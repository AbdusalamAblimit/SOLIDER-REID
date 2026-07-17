"""Full-model construction invariants for exp378 before any training run.

This check uses no dataset and no checkpoint.  It proves paired construction,
default-off shared state, initial identity modulation, and strict full-model
reload under the production model factory.
"""
import argparse
import io
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ``swin_transformer.py`` carries a legacy, unused cv2 import.  The local uv
# gate does not need image I/O; allow model construction without weakening any
# production forward check when that optional package is absent.
try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    sys.modules['cv2'] = types.ModuleType('cv2')
try:
    from mmengine.runner import load_checkpoint as _unused_loader  # noqa: F401
except ModuleNotFoundError:
    mmengine_stub = types.ModuleType('mmengine')
    mmengine_runner_stub = types.ModuleType('mmengine.runner')

    def _checkpoint_loading_is_disabled(*_args, **_kwargs):
        raise RuntimeError('model invariant gate unexpectedly loaded a checkpoint')

    mmengine_runner_stub.load_checkpoint = _checkpoint_loading_is_disabled
    mmengine_stub.runner = mmengine_runner_stub
    sys.modules['mmengine'] = mmengine_stub
    sys.modules['mmengine.runner'] = mmengine_runner_stub

from config import cfg as default_cfg
from model import make_model
from solver.make_optimizer import make_optimizer


ARMS = {
    'b0': 'configs/occluded_duke/exp378_b0_clean.yml',
    'f0': 'configs/occluded_duke/exp378_f0_frozen_anchor.yml',
    'd0': 'configs/occluded_duke/exp378_d0_continued_pose.yml',
    'p0': 'configs/occluded_duke/exp378_p0_reid_only_geometry.yml',
    'j0': 'configs/occluded_duke/exp378_j0_joint_control.yml',
    'mrf0': 'configs/occluded_duke/exp378_mrf0_sgd_relax.yml',
    'mrp0': 'configs/occluded_duke/exp378_mrp0_sgd_relax.yml',
    'r0': 'configs/occluded_duke/exp378_r0_external_teacher.yml',
    'rg0': 'configs/occluded_duke/exp378_rg0_external_gaussian.yml',
    'n0': 'configs/occluded_duke/exp378_n0_permuted_bootstrap.yml',
}

N0_PERMUTATION = tuple(range(1, 17)) + (0,)


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def arm_cfg(name):
    result = default_cfg.clone()
    result.defrost()
    result.merge_from_file(str(ROOT / ARMS[name]))
    # Construction invariants must not depend on a machine-local checkpoint.
    result.MODEL.PRETRAIN_CHOICE = 'none'
    result.MODEL.PRETRAIN_PATH = ''
    result.MODEL.WITH_CP = False
    result.freeze()
    return result


def build(name, device):
    set_seed()
    config = arm_cfg(name)
    model = make_model(
        config, num_class=702, camera_num=8, view_num=1,
        semantic_weight=config.MODEL.SEMANTIC_WEIGHT)
    return model.to(device)


def cpu_state(model):
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}


def assert_exact_state(observed, expected, label, shared_only=False):
    keys = expected.keys() & observed.keys() if shared_only else expected.keys()
    if not shared_only and observed.keys() != expected.keys():
        raise RuntimeError('%s state keys differ' % label)
    if shared_only:
        unexpected_missing = [
            key for key in expected if key not in observed
            and not key.startswith(('tapf.', 'psg_modules'))]
        if unexpected_missing:
            raise RuntimeError('%s lost shared keys: %s'
                               % (label, unexpected_missing[:5]))
    for key in keys:
        if not torch.equal(observed[key], expected[key]):
            raise RuntimeError('%s parameter differs: %s' % (label, key))


def optimizer_signature(config, model):
    center = torch.nn.Linear(1, 1)
    optimizer, _ = make_optimizer(config, model, center)
    signature = optimizer.state_dict()
    return {
        'state': signature['state'],
        'param_groups': signature['param_groups'],
        'names': tuple(name for name, parameter in model.named_parameters()
                       if parameter.requires_grad),
    }


def pose_batch(batch, device):
    torch.manual_seed(378)
    heatmaps = torch.randn(batch, 2, 17, 96, 32) * 0.01
    scores = torch.zeros(batch, 2, 17)
    person_mask = torch.zeros(batch, 2)
    person_mask[:, 0] = 1.0
    scores[:, 0] = torch.linspace(0.1, 0.9, 17)
    for sample in range(batch):
        for joint in range(17):
            y = 2 + (sample * 7 + joint * 5) % 92
            x = 1 + (sample * 3 + joint * 2) % 30
            heatmaps[sample, 0, joint, y, x] = (
                scores[sample, 0, joint] + 0.2)
    return {
        'heatmaps': heatmaps.to(device),
        'scores': scores.to(device),
        'person_mask': person_mask.to(device),
    }


def capture_first_psg_inputs(model, image, camera, view, pose_dict):
    first = model.psg_modules_dict['s3_b0']
    captured = {}

    def field_hook(_module, inputs):
        captured['field'] = inputs[2].detach().clone()

    def encoder_hook(_module, inputs):
        captured['encoder'] = inputs[0].detach().clone()

    field_handle = first.register_forward_pre_hook(field_hook)
    encoder_handle = first.encoder.register_forward_pre_hook(encoder_hook)
    try:
        with torch.no_grad():
            model(image, cam_label=camera, view_label=view,
                  pose_dict=pose_dict)
    finally:
        field_handle.remove()
        encoder_handle.remove()
    if set(captured) != {'field', 'encoder'}:
        raise RuntimeError('failed to capture PSG field/sigmoid input')
    return captured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    args = parser.parse_args()
    device = torch.device(args.device)

    b0 = build('b0', device)
    b0_state = cpu_state(b0)
    b0.eval()
    set_seed(77)
    image = torch.randn(2, 3, 384, 128, device=device)
    camera = torch.zeros(2, dtype=torch.long, device=device)
    view = torch.zeros(2, dtype=torch.long, device=device)
    with torch.no_grad():
        b0_descriptor, b0_featmaps = b0(
            image, cam_label=camera, view_label=view, pose_dict=None)
    del b0
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    p0 = build('p0', device)
    p0_state = cpu_state(p0)
    assert_exact_state(p0_state, b0_state, 'B0/P0 shared construction',
                       shared_only=True)
    p0.eval()
    p0.set_tapf_epoch(11)
    with torch.no_grad():
        p0_descriptor, p0_featmaps = p0(
            image, cam_label=camera, view_label=view, pose_dict=None)
    if not torch.equal(p0_descriptor, b0_descriptor):
        raise RuntimeError('initial zero PSG changed B0 descriptor')
    if len(p0_featmaps) != len(b0_featmaps):
        raise RuntimeError('initial zero PSG changed featmap structure')
    for index, (observed, expected) in enumerate(
            zip(p0_featmaps, b0_featmaps)):
        if not torch.equal(observed, expected):
            raise RuntimeError('initial zero PSG changed featmap %d' % index)

    payload = io.BytesIO()
    torch.save(p0.state_dict(), payload)
    payload.seek(0)
    restored = build('p0', device)
    restored.load_state_dict(
        torch.load(payload, map_location=device, weights_only=True),
        strict=True)
    restored.eval()
    restored.set_tapf_epoch(11)
    with torch.no_grad():
        restored_descriptor, restored_featmaps = restored(
            image, cam_label=camera, view_label=view, pose_dict=None)
    if not torch.equal(restored_descriptor, p0_descriptor):
        raise RuntimeError('strict reload changed P0 descriptor')
    for index, (observed, expected) in enumerate(
            zip(restored_featmaps, p0_featmaps)):
        if not torch.equal(observed, expected):
            raise RuntimeError('strict reload changed P0 featmap %d' % index)
    del restored

    reference = p0_state
    del p0, p0_state
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    for name in ('f0', 'd0', 'j0', 'mrf0', 'mrp0'):
        candidate = build(name, device)
        candidate_state = cpu_state(candidate)
        assert_exact_state(candidate_state, reference,
                           'P0/%s paired init' % name.upper())
        del candidate, candidate_state
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # N0 is hard F0 with only a fixed bootstrap joint relabeling and output
    # directory change.  The relabeling is immutable metadata: construction
    # RNG, model state, parameters and optimizer groups must remain exact.
    f0_config = arm_cfg('f0')
    n0_config = arm_cfg('n0')
    normalized_n0 = n0_config.clone()
    normalized_n0.defrost()
    normalized_n0.MODEL.POSE_TAPF_BOOTSTRAP_JOINT_PERMUTATION = []
    normalized_n0.OUTPUT_DIR = f0_config.OUTPUT_DIR
    normalized_n0.freeze()
    if normalized_n0.dump() != f0_config.dump():
        raise RuntimeError('N0 config differs from hard F0 beyond permutation/output')
    if tuple(n0_config.MODEL.POSE_TAPF_BOOTSTRAP_JOINT_PERMUTATION) \
            != N0_PERMUTATION:
        raise RuntimeError('N0 config permutation differs from preregistration')

    external_pose = pose_batch(image.shape[0], device)
    target_raw = (external_pose['heatmaps'][:, 0]
                  * external_pose['person_mask'][:, 0, None, None, None])
    f0 = build('f0', device)
    f0_rng = torch.get_rng_state().clone()
    f0_cuda_rng = (torch.cuda.get_rng_state_all()
                   if device.type == 'cuda' else None)
    f0_state = cpu_state(f0)
    f0_optimizer = optimizer_signature(f0_config, f0)
    f0.train()
    f0.set_tapf_epoch(1)
    set_seed(3780)
    f0_capture = capture_first_psg_inputs(
        f0, image, camera, view, external_pose)
    if not torch.equal(f0_capture['field'], target_raw):
        raise RuntimeError('hard F0 epoch-1 field is not exact teacher')
    del f0
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    n0 = build('n0', device)
    n0_rng = torch.get_rng_state().clone()
    n0_cuda_rng = (torch.cuda.get_rng_state_all()
                   if device.type == 'cuda' else None)
    n0_state = cpu_state(n0)
    assert_exact_state(n0_state, f0_state, 'F0/N0 permutation-only init')
    if not torch.equal(n0_rng, f0_rng):
        raise RuntimeError('N0 changed CPU construction RNG')
    if f0_cuda_rng is not None:
        for observed, expected in zip(n0_cuda_rng, f0_cuda_rng):
            if not torch.equal(observed, expected):
                raise RuntimeError('N0 changed CUDA construction RNG')
    n0_optimizer = optimizer_signature(n0_config, n0)
    if n0_optimizer != f0_optimizer:
        raise RuntimeError('F0/N0 optimizer parameter groups differ')
    if tuple(n0.tapf.bootstrap_joint_permutation) != N0_PERMUTATION:
        raise RuntimeError('N0 full model lost fixed permutation metadata')
    n0.train()
    n0.set_tapf_epoch(1)
    set_seed(3780)
    n0_capture = capture_first_psg_inputs(
        n0, image, camera, view, external_pose)
    permutation_index = torch.tensor(
        N0_PERMUTATION, dtype=torch.long, device=device)
    expected_n0_field = target_raw.index_select(1, permutation_index)
    if not torch.equal(n0_capture['field'], expected_n0_field):
        raise RuntimeError('N0 PSG input is not exactly once-permuted teacher')
    if torch.equal(n0_capture['field'], target_raw):
        raise RuntimeError('N0 permutation was inactive or double-inverted')
    del n0
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # R0/RG0 must be a true renderer-only pair.  The RG0 renderer has no
    # persistent state or parameters, so construction RNG, model state and
    # optimizer groups remain exact while only the field passed to PSG differs.
    r0_config = arm_cfg('r0')
    r0 = build('r0', device)
    r0_rng = torch.get_rng_state().clone()
    r0_cuda_rng = (torch.cuda.get_rng_state_all()
                   if device.type == 'cuda' else None)
    r0_state = cpu_state(r0)
    r0_optimizer = optimizer_signature(r0_config, r0)
    r0.eval()
    external_pose = pose_batch(image.shape[0], device)
    r0_capture = capture_first_psg_inputs(
        r0, image, camera, view, external_pose)
    expected_raw = (external_pose['heatmaps'][:, 0]
                    * external_pose['person_mask'][:, 0, None, None, None])
    if not torch.equal(r0_capture['field'], expected_raw):
        raise RuntimeError('R0 PSG input is not exact target-person raw heatmap')
    expected_raw_at_psg = torch.nn.functional.interpolate(
        expected_raw, size=r0_capture['encoder'].shape[-2:],
        mode='bilinear', align_corners=False)
    if not torch.equal(
            r0_capture['encoder'], torch.sigmoid(expected_raw_at_psg)):
        raise RuntimeError('R0 PSG sigmoid boundary changed')
    del r0
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    rg0_config = arm_cfg('rg0')
    rg0 = build('rg0', device)
    rg0_rng = torch.get_rng_state().clone()
    rg0_cuda_rng = (torch.cuda.get_rng_state_all()
                    if device.type == 'cuda' else None)
    rg0_state = cpu_state(rg0)
    assert_exact_state(rg0_state, r0_state, 'R0/RG0 renderer-only init')
    if not torch.equal(rg0_rng, r0_rng):
        raise RuntimeError('RG0 changed CPU construction RNG')
    if r0_cuda_rng is not None:
        for observed, expected in zip(rg0_cuda_rng, r0_cuda_rng):
            if not torch.equal(observed, expected):
                raise RuntimeError('RG0 changed CUDA construction RNG')
    rg0_optimizer = optimizer_signature(rg0_config, rg0)
    if rg0_optimizer != r0_optimizer:
        raise RuntimeError('R0/RG0 optimizer parameter groups differ')
    if list(rg0.external_field_renderer.parameters()):
        raise RuntimeError('RG0 renderer unexpectedly has parameters')
    if rg0.external_field_renderer.state_dict():
        raise RuntimeError('RG0 renderer unexpectedly has persistent state')
    rg0.eval()
    rg0_capture = capture_first_psg_inputs(
        rg0, image, camera, view, external_pose)
    expected_rendered, _ = rg0.external_field_renderer(
        expected_raw,
        external_pose['scores'][:, 0]
        * external_pose['person_mask'][:, 0:1])
    if not torch.equal(rg0_capture['field'], expected_rendered):
        raise RuntimeError('RG0 PSG input differs from shared renderer output')
    expected_rendered_at_psg = torch.nn.functional.interpolate(
        expected_rendered, size=rg0_capture['encoder'].shape[-2:],
        mode='bilinear', align_corners=False)
    if not torch.equal(
            rg0_capture['encoder'], torch.sigmoid(expected_rendered_at_psg)):
        raise RuntimeError('RG0 PSG must apply exactly one sigmoid at boundary')

    print('TAPF_MODEL_INVARIANTS_PASS')
    print('RG0_MODEL_INVARIANTS_PASS')
    print('N0_MODEL_INVARIANTS_PASS')
    print('device=%s b0_keys=%d tapf_keys=%d r0_keys=%d descriptor=%s featmaps=%d'
          % (device, len(b0_state), len(reference),
             len(r0_state), tuple(b0_descriptor.shape), len(b0_featmaps)))


if __name__ == '__main__':
    main()
