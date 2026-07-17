"""Full-model and RGB-only evaluator invariants for exp383.

This gate is intentionally training-free. It builds the exact Market B0/D0
models, checks paired initialization/state/RNG/optimizer semantics, exercises
the D0 epoch-1 and epoch-11 routes, and reads a real Occluded-ReID RGB batch
without requiring pose files. CUDA/AMP/overflow and ten-step parity are a
separate production gate and are not implied by this script.
"""

import argparse
import io
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import cfg as root_cfg
from datasets.bases import ImageDataset
from model import make_model
from solver import make_optimizer
from test_on_occluded_reid import (
    _build_val_loader,
    _extract_features,
    _uses_external_pose_at_eval,
)


CONFIGS = {
    'b0': ROOT / 'configs' / 'market' / 'exp383_b0.yml',
    'd0': ROOT / 'configs' / 'market' / 'exp383_d0.yml',
}


class ExplodingPoseDict(dict):
    def __getitem__(self, key):
        raise RuntimeError('eval touched external pose: ' + key)

    def get(self, key, default=None):
        del default
        raise RuntimeError('eval touched external pose: ' + key)


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(arm, load_pretrained=False):
    config = root_cfg.clone()
    config.defrost()
    config.merge_from_file(str(CONFIGS[arm]))
    if not load_pretrained:
        config.MODEL.PRETRAIN_CHOICE = 'none'
        config.MODEL.PRETRAIN_PATH = ''
    config.MODEL.WITH_CP = False
    config.DATALOADER.NUM_WORKERS = 0
    config.freeze()
    return config


def build(arm, device):
    config = load_config(arm)
    set_seed(int(config.SOLVER.SEED))
    model = make_model(
        config, num_class=751, camera_num=6, view_num=1,
        semantic_weight=config.MODEL.SEMANTIC_WEIGHT,
    ).to(device)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = (torch.cuda.get_rng_state_all()
                if device.type == 'cuda' else None)
    return config, model, cpu_rng, cuda_rng


def synthetic_input(device, batch=2):
    generator = torch.Generator().manual_seed(383)
    image = torch.randn(batch, 3, 384, 128, generator=generator).to(device)
    # Production pose tensors always reserve at least person-0 (target) and
    # person-1 (distractor).  Keep a real distractor here so _prepare_pose's
    # non-target max-reduction is exercised instead of receiving an empty
    # person dimension.
    heatmaps = torch.zeros(batch, 2, 17, 96, 32)
    scores = torch.linspace(0.15, 0.95, 17).repeat(batch, 2, 1)
    for sample in range(batch):
        for joint in range(17):
            y = 2 + (sample * 7 + joint * 5) % 92
            x = 1 + (sample * 3 + joint * 2) % 30
            heatmaps[sample, 0, joint, y, x] = scores[sample, 0, joint]
            distractor_y = 2 + (sample * 11 + joint * 3) % 92
            distractor_x = 1 + (sample * 5 + joint * 7) % 30
            heatmaps[sample, 1, joint, distractor_y, distractor_x] = \
                scores[sample, 1, joint]
    pose = {
        'heatmaps': heatmaps.to(device),
        'scores': scores.to(device),
        'person_mask': torch.ones(batch, 2, device=device),
    }
    camera = torch.zeros(batch, dtype=torch.long, device=device)
    view = torch.ones(batch, dtype=torch.long, device=device)
    label = torch.arange(batch, dtype=torch.long, device=device)
    return image, pose, camera, view, label


def assert_finite_state(model, label):
    for name, value in model.state_dict().items():
        if not bool(torch.isfinite(value.detach().float()).all()):
            raise RuntimeError('%s has non-finite state: %s' % (label, name))


def grad_norm(objective, parameters):
    parameters = [parameter for parameter in parameters
                  if parameter.requires_grad]
    if objective is None or not objective.requires_grad or not parameters:
        return 0.0
    gradients = torch.autograd.grad(
        objective, parameters, retain_graph=True, allow_unused=True)
    total = 0.0
    for gradient in gradients:
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError('non-finite gradient')
        total += float(gradient.detach().float().square().sum().item())
    return math.sqrt(total)


def optimizer_options(config, model):
    center = torch.nn.Linear(1, 1)
    optimizer, _ = make_optimizer(config, model, center)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    result = {}
    for group in optimizer.param_groups:
        if len(group['params']) != 1:
            raise RuntimeError('optimizer group is not per-parameter')
        parameter = group['params'][0]
        result[names[id(parameter)]] = (
            float(group['lr']), float(group['weight_decay']))
    return result


def capture_first_field(model, image, pose, camera, view, label, epoch):
    captured = []
    first_psg = model.psg_modules_dict['s3_b0']
    handle = first_psg.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[2].detach().clone()))
    model.train()
    model.set_tapf_epoch(epoch)
    try:
        output = model(
            image, label=label, cam_label=camera, view_label=view,
            pose_dict=pose)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError('did not capture exactly one first PSG field')
    return captured[0], output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    parser.add_argument('--occluded-root', default='data/occluded_reid')
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')

    b0_cfg = load_config('b0')
    d0_cfg = load_config('d0')
    for section in ('INPUT', 'DATASETS', 'DATALOADER', 'SOLVER', 'TEST'):
        if b0_cfg[section] != d0_cfg[section]:
            raise RuntimeError('unmatched config section: ' + section)
    if (b0_cfg.INPUT.RE_PROB != 0.5
            or b0_cfg.SOLVER.IMS_PER_BATCH != 64
            or b0_cfg.SOLVER.SEED != 1234
            or b0_cfg.TEST.FLIP_TEST):
        raise RuntimeError('exp383 fixed recipe contract failed')
    if b0_cfg.MODEL.POSE_ENABLED:
        raise RuntimeError('B0 must remain RGB-only')
    if not d0_cfg.MODEL.POSE_ENABLED or not d0_cfg.MODEL.POSE_TAPF:
        raise RuntimeError('D0 must enable atomic TAPF')
    if _uses_external_pose_at_eval(d0_cfg):
        raise RuntimeError('D0 evaluator still requests external pose')

    b0_cfg, b0, b0_rng, b0_cuda_rng = build('b0', device)
    d0_cfg, d0, d0_rng, d0_cuda_rng = build('d0', device)
    if not torch.equal(b0_rng, d0_rng):
        raise RuntimeError('D0 construction changed the CPU RNG stream')
    if b0_cuda_rng is not None:
        for expected, observed in zip(b0_cuda_rng, d0_cuda_rng):
            if not torch.equal(expected, observed):
                raise RuntimeError('D0 construction changed CUDA RNG stream')

    b0_state = b0.state_dict()
    d0_state = d0.state_dict()
    for key, expected in b0_state.items():
        if key not in d0_state:
            raise RuntimeError('D0 lost shared B0 state: ' + key)
        if not torch.equal(expected, d0_state[key]):
            raise RuntimeError('D0 changed shared initialization: ' + key)
    assert_finite_state(b0, 'B0')
    assert_finite_state(d0, 'D0')

    b0_optimizer = optimizer_options(b0_cfg, b0)
    d0_optimizer = optimizer_options(d0_cfg, d0)
    for name, expected in b0_optimizer.items():
        if d0_optimizer.get(name) != expected:
            raise RuntimeError('D0 changed shared optimizer option: ' + name)

    image, pose, camera, view, label = synthetic_input(device)
    b0.eval()
    d0.eval()
    d0.set_tapf_epoch(11)
    with torch.no_grad():
        b0_descriptor = b0(
            image, cam_label=camera, view_label=view)[0]
        d0_descriptor = d0(
            image, cam_label=camera, view_label=view, pose_dict=None)[0]
    if not torch.equal(b0_descriptor, d0_descriptor):
        raise RuntimeError('zero-init D0 does not reproduce B0 descriptor')

    payload = io.BytesIO()
    torch.save(d0.state_dict(), payload)
    payload.seek(0)
    _, replica, _, _ = build('d0', device)
    restored = torch.load(payload, map_location=device)
    result = replica.load_state_dict(restored, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError('D0 strict reload failed')

    epoch1_field, epoch1_output = capture_first_field(
        d0, image, pose, camera, view, label, epoch=1)
    expected_teacher = pose['heatmaps'][:, 0]
    if not torch.equal(epoch1_field, expected_teacher):
        raise RuntimeError('D0 epoch-1 PSG field is not exact teacher')
    if epoch1_output[4]['tapf_pose_loss'] is None:
        raise RuntimeError('D0 epoch-1 pose loss missing')

    epoch11_field, epoch11_output = capture_first_field(
        d0, image, pose, camera, view, label, epoch=11)
    if torch.equal(epoch11_field, expected_teacher):
        raise RuntimeError('D0 epoch-11 field did not hand off to student')
    score, feature, _, _, data = epoch11_output
    pose_loss = data['tapf_pose_loss']
    if pose_loss is None:
        raise RuntimeError('D0 epoch-11 continued pose loss missing')
    reid_proxy = score.float().square().mean() + feature.float().square().mean()
    anchor = list(d0.tapf.anchor.parameters())
    adapter = list(d0.tapf.geometry_adapter.parameters())
    psg = list(d0.psg_modules_dict.parameters())
    backbone = [d0.base.patch_embed.projection.weight]
    ownership = {
        'pose_anchor': grad_norm(pose_loss, anchor),
        'pose_adapter': grad_norm(pose_loss, adapter),
        'pose_psg': grad_norm(pose_loss, psg),
        'pose_backbone': grad_norm(pose_loss, backbone),
        'reid_anchor': grad_norm(reid_proxy, anchor),
        'reid_adapter': grad_norm(reid_proxy, adapter),
        'reid_psg': grad_norm(reid_proxy, psg),
        'reid_backbone': grad_norm(reid_proxy, backbone),
    }
    if ownership['pose_anchor'] <= 0:
        raise RuntimeError('pose loss missed anchor')
    if any(ownership[key] != 0 for key in (
            'pose_adapter', 'pose_psg', 'pose_backbone',
            'reid_anchor', 'reid_adapter')):
        raise RuntimeError('D0 gradient ownership leaked: %r' % ownership)
    if ownership['reid_psg'] <= 0 or ownership['reid_backbone'] <= 0:
        raise RuntimeError('ReID loss missed PSG/backbone: %r' % ownership)

    shuffled = {key: value.flip(0) for key, value in pose.items()}
    d0.eval()
    with torch.no_grad():
        descriptors = {
            'correct': d0(image, cam_label=camera, view_label=view,
                          pose_dict=pose)[0],
            'shuffle': d0(image, cam_label=camera, view_label=view,
                          pose_dict=shuffled)[0],
            'none': d0(image, cam_label=camera, view_label=view,
                       pose_dict=None)[0],
            'exploding': d0(image, cam_label=camera, view_label=view,
                            pose_dict=ExplodingPoseDict())[0],
        }
    for name, observed in descriptors.items():
        if not torch.equal(observed, descriptors['correct']):
            raise RuntimeError('D0 eval pose parity failed: ' + name)

    loader_cfg = d0_cfg.clone()
    loader_cfg.defrost()
    loader_cfg.DATALOADER.NUM_WORKERS = 0
    loader_cfg.TEST.IMS_PER_BATCH = 64
    loader_cfg.freeze()
    dataset, loader = _build_val_loader(loader_cfg, args.occluded_root)
    if len(dataset.query) != 1000 or len(dataset.gallery) != 1000:
        raise RuntimeError('unexpected Occluded-ReID split sizes')
    if not all(isinstance(part, ImageDataset)
               for part in loader.dataset.datasets):
        raise RuntimeError('TAPF evaluator did not use ordinary RGB datasets')
    batch = next(iter(loader))
    if len(batch) != 6 or batch[0].shape[0] != 64:
        raise RuntimeError('TAPF evaluator did not produce RGB batch64')
    rgb_image, _, _, rgb_camera, rgb_view, _ = batch
    rgb_image = rgb_image[:2].to(device)
    rgb_camera = rgb_camera[:2].to(device)
    rgb_view = rgb_view[:2].to(device)
    with torch.no_grad():
        extracted = _extract_features(
            d0, loader_cfg, rgb_image, rgb_camera, rgb_view,
            ExplodingPoseDict(), 'global')
    if extracted.shape[0] != 2 or not bool(torch.isfinite(extracted).all()):
        raise RuntimeError('RGB-only evaluator descriptor invalid')

    legacy_cfg = loader_cfg.clone()
    legacy_cfg.defrost()
    legacy_cfg.MODEL.POSE_TAPF = False
    legacy_cfg.freeze()
    try:
        _build_val_loader(legacy_cfg, args.occluded_root)
    except RuntimeError as error:
        if 'requires Occluded-ReID pose data' not in str(error):
            raise
    else:
        raise RuntimeError('legacy evaluator silently accepted missing pose')

    total_b0 = sum(parameter.numel() for parameter in b0.parameters())
    total_d0 = sum(parameter.numel() for parameter in d0.parameters())
    dedicated = sum(parameter.numel() for name, parameter
                    in d0.named_parameters()
                    if name.startswith(('tapf.', 'psg_modules_dict.')))
    print('EXP383_FULL_MODEL_PREFLIGHT_PASS')
    print('device=%s b0_params=%d d0_params=%d dedicated=%d overhead=%d'
          % (device, total_b0, total_d0, dedicated, total_d0 - total_b0))
    print('shared_state=%d d0_state=%d optimizer_shared=%d'
          % (len(b0_state), len(d0_state), len(b0_optimizer)))
    print('gradient_ownership=%r' % ownership)
    print('occluded_reid=query%d/gallery%d rgb_batch=%s pose_free=exact'
          % (len(dataset.query), len(dataset.gallery), tuple(batch[0].shape)))


if __name__ == '__main__':
    main()
