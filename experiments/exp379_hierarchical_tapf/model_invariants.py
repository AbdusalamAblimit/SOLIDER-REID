"""Full-model CPU invariants for exp379 before any CUDA training gate."""
import io
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    sys.modules['cv2'] = types.ModuleType('cv2')
try:
    from mmengine.runner import load_checkpoint as _unused_loader  # noqa: F401
except ModuleNotFoundError:
    mmengine_stub = types.ModuleType('mmengine')
    runner_stub = types.ModuleType('mmengine.runner')

    def _checkpoint_loading_is_disabled(*_args, **_kwargs):
        raise RuntimeError('model invariant unexpectedly loaded a checkpoint')

    runner_stub.load_checkpoint = _checkpoint_loading_is_disabled
    mmengine_stub.runner = runner_stub
    sys.modules['mmengine'] = mmengine_stub
    sys.modules['mmengine.runner'] = runner_stub

from config import cfg as default_cfg
from model import make_model
from solver.make_optimizer import make_optimizer


CONFIGS = {
    'b0': 'configs/occluded_duke/exp378_b0_clean.yml',
    'd0': 'configs/occluded_duke/exp378_d0_continued_pose.yml',
    'ht0': 'configs/occluded_duke/exp379_ht0_hierarchical_tapf.yml',
}


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cfg(name):
    result = default_cfg.clone()
    result.defrost()
    result.merge_from_file(str(ROOT / CONFIGS[name]))
    result.MODEL.PRETRAIN_CHOICE = 'none'
    result.MODEL.PRETRAIN_PATH = ''
    result.MODEL.WITH_CP = False
    result.freeze()
    return result


def build(name):
    set_seed()
    config = load_cfg(name)
    model = make_model(
        config, num_class=702, camera_num=8, view_num=1,
        semantic_weight=config.MODEL.SEMANTIC_WEIGHT)
    return config, model


def cpu_state(model):
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}


def assert_shared_exact(left, right, label, skip_prefixes):
    keys = [key for key in left if not key.startswith(skip_prefixes)]
    missing = [key for key in keys if key not in right]
    if missing:
        raise RuntimeError('%s missing shared keys: %s'
                           % (label, missing[:5]))
    for key in keys:
        if not torch.equal(left[key], right[key]):
            raise RuntimeError('%s differs at %s' % (label, key))


def optimizer_ids(config, model):
    center = torch.nn.Linear(1, 1)
    optimizer, _ = make_optimizer(config, model, center)
    ids = [id(parameter) for group in optimizer.param_groups
           for parameter in group['params']]
    if len(ids) != len(set(ids)):
        raise RuntimeError('optimizer contains duplicate parameters')
    expected = {id(parameter) for parameter in model.parameters()
                if parameter.requires_grad}
    if set(ids) != expected:
        raise RuntimeError('optimizer does not cover every trainable parameter')
    return optimizer


class ExplodingPoseDict(dict):
    def __getitem__(self, key):
        raise RuntimeError('RGB-only eval touched external pose: ' + key)

    def get(self, key, default=None):
        raise RuntimeError('RGB-only eval touched external pose: ' + key)


def capture_fields(model, image, camera, view, pose_dict=None):
    captured = {}
    handles = []
    for key in ('s2_b0', 's3_b0'):
        def hook(_module, inputs, key=key):
            captured[key] = inputs[2].detach().float().cpu().clone()
        handles.append(model.psg_modules_dict[key].register_forward_pre_hook(
            hook))
    try:
        output = model(
            image, cam_label=camera, view_label=view, pose_dict=pose_dict)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != {'s2_b0', 's3_b0'}:
        raise RuntimeError('failed to capture both hierarchical PSG fields')
    return output, captured


def pose_batch(batch):
    torch.manual_seed(379)
    heatmaps = torch.randn(batch, 2, 17, 96, 32) * 0.005
    scores = torch.zeros(batch, 2, 17)
    person_mask = torch.zeros(batch, 2)
    person_mask[:, 0] = 1.0
    scores[:, 0] = torch.linspace(0.15, 0.95, 17)
    for sample in range(batch):
        for joint in range(17):
            y = 2 + (sample * 7 + joint * 5) % 92
            x = 1 + (sample * 3 + joint * 2) % 30
            heatmaps[sample, 0, joint, y, x] = (
                scores[sample, 0, joint] + 0.15)
    return {
        'heatmaps': heatmaps,
        'scores': scores,
        'person_mask': person_mask,
    }


def nonzero_objective_grad(objective, parameters):
    parameters = [parameter for parameter in parameters
                  if parameter.requires_grad]
    gradients = torch.autograd.grad(
        objective, parameters, retain_graph=True, allow_unused=True)
    return any(gradient is not None
               and bool(torch.isfinite(gradient).all())
               and float(gradient.abs().sum()) > 0.0
               for gradient in gradients)


def main():
    _, b0 = build('b0')
    b0_rng = torch.get_rng_state().clone()
    b0_state = cpu_state(b0)
    d0_config, d0 = build('d0')
    d0_rng = torch.get_rng_state().clone()
    d0_state = cpu_state(d0)
    ht0_config, ht0 = build('ht0')
    ht0_rng = torch.get_rng_state().clone()
    ht0_state = cpu_state(ht0)

    if not torch.equal(b0_rng, d0_rng) or not torch.equal(d0_rng, ht0_rng):
        raise RuntimeError('B0/D0/HT0 construction RNG differs')
    assert_shared_exact(
        d0_state, b0_state, 'D0/B0 shared init',
        skip_prefixes=('tapf.', 'psg_modules'))
    assert_shared_exact(
        d0_state, ht0_state, 'D0/HT0 shared init',
        skip_prefixes=('tapf.', 'psg_modules_dict.s2_'))
    for key in d0_state:
        if key.startswith('psg_modules_dict.s3_'):
            if not torch.equal(d0_state[key], ht0_state[key]):
                raise RuntimeError('shared Stage-3 PSG init differs: ' + key)

    if not ht0.use_hierarchical_tapf:
        raise RuntimeError('HT0 did not activate hierarchical TAPF')
    if ht0.tapf.source_stages != (1, 2):
        raise RuntimeError('unexpected HT0 source stages')
    if ht0.psg_stage_indices != {2, 3}:
        raise RuntimeError('unexpected HT0 consumer stages')
    if set(ht0.tapf.stage_projections.keys()) != {'1', '2'}:
        raise RuntimeError('unexpected HT0 stage projections')
    if sum(name == 'anchor' for name, _ in ht0.tapf.named_modules()) != 1:
        raise RuntimeError('HT0 does not have exactly one shared decoder')
    optimizer_ids(d0_config, d0)
    optimizer_ids(ht0_config, ht0)

    set_seed(77)
    image = torch.randn(2, 3, 384, 128)
    camera = torch.zeros(2, dtype=torch.long)
    view = torch.zeros(2, dtype=torch.long)
    b0.eval()
    d0.eval()
    ht0.eval()
    d0.set_tapf_epoch(11)
    ht0.set_tapf_epoch(11)
    with torch.no_grad():
        b0_descriptor, b0_featmaps = b0(
            image, cam_label=camera, view_label=view, pose_dict=None)
        d0_descriptor, d0_featmaps = d0(
            image, cam_label=camera, view_label=view, pose_dict=None)
        (ht0_descriptor, ht0_featmaps), fields = capture_fields(
            ht0, image, camera, view, pose_dict=ExplodingPoseDict())
    for label, descriptor in (
            ('D0', d0_descriptor), ('HT0', ht0_descriptor)):
        if not torch.equal(descriptor, b0_descriptor):
            raise RuntimeError(label + ' zero-init descriptor differs from B0')
    for label, featmaps in (('D0', d0_featmaps), ('HT0', ht0_featmaps)):
        for index, (observed, expected) in enumerate(
                zip(featmaps, b0_featmaps)):
            if not torch.equal(observed, expected):
                raise RuntimeError('%s zero-init featmap %d differs from B0'
                                   % (label, index))
    if torch.equal(fields['s2_b0'], fields['s3_b0']):
        raise RuntimeError('HT0 deeper field did not refine the shallower field')
    if float(ht0._last_tapf_data['tapf_stats'][
            'hierarchical_stage_count']) != 2.0:
        raise RuntimeError('HT0 did not aggregate both source stages')

    payload = io.BytesIO()
    torch.save(ht0.state_dict(), payload)
    payload.seek(0)
    _, restored = build('ht0')
    restored.load_state_dict(
        torch.load(payload, map_location='cpu', weights_only=True),
        strict=True)
    restored.eval()
    restored.set_tapf_epoch(11)
    with torch.no_grad():
        restored_descriptor, restored_featmaps = restored(
            image, cam_label=camera, view_label=view, pose_dict=None)
    if not torch.equal(restored_descriptor, ht0_descriptor):
        raise RuntimeError('HT0 strict reload changed descriptor')
    for index, (observed, expected) in enumerate(
            zip(restored_featmaps, ht0_featmaps)):
        if not torch.equal(observed, expected):
            raise RuntimeError('HT0 strict reload changed featmap %d' % index)

    ht0.train()
    ht0.set_tapf_epoch(1)
    pose = pose_batch(image.shape[0])
    output, train_fields = capture_fields(
        ht0, image, camera, view, pose_dict=pose)
    cls_score, global_feat, _, _, data = output
    target_teacher = pose['heatmaps'][:, 0]
    torch.testing.assert_close(
        train_fields['s2_b0'], target_teacher, rtol=0, atol=0)
    torch.testing.assert_close(
        train_fields['s3_b0'], target_teacher, rtol=0, atol=0)
    pose_loss = data['tapf_pose_loss']
    if pose_loss is None or not bool(torch.isfinite(pose_loss)):
        raise RuntimeError('HT0 pose loss is missing/non-finite')
    if not nonzero_objective_grad(
            pose_loss, ht0.tapf.stage_projections.parameters()):
        raise RuntimeError('HT0 pose loss did not update stage projections')
    if not nonzero_objective_grad(pose_loss, ht0.tapf.anchor.parameters()):
        raise RuntimeError('HT0 pose loss did not update shared decoder')
    if nonzero_objective_grad(pose_loss, ht0.base.parameters()):
        raise RuntimeError('HT0 pose loss leaked into backbone')

    reid_objective = cls_score.float().sum() + global_feat.float().sum()
    if nonzero_objective_grad(reid_objective, ht0.tapf.parameters()):
        raise RuntimeError('HT0 ReID objective leaked into pose module')
    if not nonzero_objective_grad(
            reid_objective, ht0.psg_modules_dict.parameters()):
        raise RuntimeError('HT0 ReID objective did not reach PSG')

    print('EXP379_MODEL_INVARIANTS_PASS')
    print('b0_keys=%d d0_keys=%d ht0_keys=%d ht0_params=%d '
          'field_delta=%.9f'
          % (len(b0_state), len(d0_state), len(ht0_state),
             sum(parameter.numel() for parameter in ht0.tapf.parameters()),
             float((fields['s2_b0'] - fields['s3_b0']).abs().max())))


if __name__ == '__main__':
    main()
