"""CPU unit and model-invariant gates for exp380 ResNet TAPF."""
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import cfg as root_cfg
from model import make_model
from model.backbones.resnet import torch_load_compat
from model.make_model import Backbone


CONFIGS = {
    'b0': 'configs/occluded_duke/exp380_r50_b0.yml',
    'd0': 'configs/occluded_duke/exp380_r50_d0.yml',
    'ht0': 'configs/occluded_duke/exp380_r50_ht0.yml',
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


def load_config(name):
    config = root_cfg.clone()
    config.defrost()
    config.merge_from_file(str(ROOT / CONFIGS[name]))
    # Unit gates validate initialization and routing without requiring the
    # external ImageNet checkpoint.  Production gates restore the real path.
    config.MODEL.PRETRAIN_CHOICE = 'none'
    config.MODEL.PRETRAIN_PATH = ''
    config.DATALOADER.NUM_WORKERS = 0
    config.freeze()
    return config


def build(name):
    config = load_config(name)
    set_seed(int(config.SOLVER.SEED))
    model = make_model(
        config, num_class=7, camera_num=0, view_num=0,
        semantic_weight=config.MODEL.SEMANTIC_WEIGHT,
    )
    return config, model, torch.get_rng_state().clone()


def synthetic_batch(batch=2):
    generator = torch.Generator().manual_seed(321)
    image = torch.randn(batch, 3, 64, 32, generator=generator)
    heatmaps = torch.rand(
        batch, 1, 17, 96, 32, generator=generator)
    scores = torch.rand(batch, 1, 17, generator=generator)
    person_mask = torch.ones(batch, 1)
    pose = {
        'heatmaps': heatmaps,
        'scores': scores,
        'person_mask': person_mask,
    }
    return image, pose


def tensor_grad_norm(objective, parameters):
    parameters = [parameter for parameter in parameters
                  if parameter.requires_grad]
    gradients = torch.autograd.grad(
        objective, parameters, retain_graph=True, allow_unused=True)
    total = 0.0
    for gradient in gradients:
        if gradient is not None:
            if not bool(torch.isfinite(gradient).all()):
                raise AssertionError('non-finite gradient')
            total += float(gradient.detach().float().square().sum())
    return total ** 0.5


def test_config_contract():
    b0 = load_config('b0')
    d0 = load_config('d0')
    ht0 = load_config('ht0')
    assert b0.MODEL.NAME == d0.MODEL.NAME == ht0.MODEL.NAME == 'resnet50'
    assert not b0.MODEL.POSE_ENABLED
    assert d0.MODEL.POSE_ENABLED and ht0.MODEL.POSE_ENABLED
    assert not d0.MODEL.POSE_TAPF_HIERARCHICAL
    assert ht0.MODEL.POSE_TAPF_HIERARCHICAL
    assert tuple(d0.MODEL.POSE_PSG_STAGES) == (3,)
    assert tuple(ht0.MODEL.POSE_PSG_STAGES) == (2, 3)
    for key in ('SOLVER', 'INPUT', 'DATASETS', 'DATALOADER', 'TEST'):
        assert b0[key] == d0[key] == ht0[key]


def test_torch_load_compatibility():
    original = torch.load
    calls = []

    def legacy_load(path, **kwargs):
        calls.append((path, dict(kwargs)))
        if 'weights_only' in kwargs:
            raise TypeError("unexpected keyword argument 'weights_only'")
        return {'ok': True}

    torch.load = legacy_load
    try:
        result = torch_load_compat('trusted-local-checkpoint.pth')
    finally:
        torch.load = original
    assert result == {'ok': True}
    assert calls == [
        ('trusted-local-checkpoint.pth', {'weights_only': False}),
        ('trusted-local-checkpoint.pth', {}),
    ]


def test_b0_existing_path_exact():
    config = load_config('b0')
    set_seed()
    expected = Backbone(7, config)
    expected_rng = torch.get_rng_state().clone()
    set_seed()
    observed = make_model(
        config, num_class=7, camera_num=0, view_num=0,
        semantic_weight=config.MODEL.SEMANTIC_WEIGHT)
    observed_rng = torch.get_rng_state().clone()
    assert type(observed) is Backbone
    assert tuple(expected.state_dict()) == tuple(observed.state_dict())
    for key, value in expected.state_dict().items():
        assert torch.equal(value, observed.state_dict()[key]), key
    assert torch.equal(expected_rng, observed_rng)


def test_matched_initialization_and_topology():
    _, b0, b0_rng = build('b0')
    _, d0, d0_rng = build('d0')
    _, ht0, ht0_rng = build('ht0')
    assert d0.psg_stage_indices == {3}
    assert ht0.psg_stage_indices == {2, 3}
    assert d0.tapf_source_stage == 2
    assert ht0.tapf_source_stages == (1, 2)
    assert ht0.tapf.source_stages == (1, 2)
    assert len(ht0.tapf.stage_projections) == 2
    assert sum(1 for _ in ht0.tapf.anchor.modules()) > 1

    for prefix in ('base.', 'bottleneck.', 'classifier.'):
        keys = [key for key in b0.state_dict() if key.startswith(prefix)]
        assert keys
        for key in keys:
            assert torch.equal(b0.state_dict()[key], d0.state_dict()[key]), key
            assert torch.equal(b0.state_dict()[key], ht0.state_dict()[key]), key
    d0_psg = d0.psg_modules_dict.state_dict()
    ht0_psg = ht0.psg_modules_dict.state_dict()
    stage3_keys = [key for key in d0_psg if key.startswith('s3_')]
    assert stage3_keys and all(key in ht0_psg for key in stage3_keys)
    for key in stage3_keys:
        assert torch.equal(d0_psg[key], ht0_psg[key]), key
    assert torch.equal(b0_rng, d0_rng)
    assert torch.equal(b0_rng, ht0_rng)


def capture_fields(model, image, pose, epoch):
    captured = {}
    handles = []
    for stage in sorted(model.psg_stage_indices):
        key = 's%d_b0' % stage

        def hook(_module, inputs, stage=stage):
            captured[stage] = inputs[2].detach().clone()

        handles.append(
            model.psg_modules_dict[key].register_forward_pre_hook(hook))
    model.train()
    model.set_tapf_epoch(epoch)
    try:
        with torch.no_grad():
            output = model(image, label=torch.zeros(
                image.shape[0], dtype=torch.long), pose_dict=pose)
    finally:
        for handle in handles:
            handle.remove()
    return captured, output


def test_e1_e11_progressive_routes():
    _, ht0, _ = build('ht0')
    image, pose = synthetic_batch()
    teacher = pose['heatmaps'][:, 0]
    e1_fields, e1_output = capture_fields(ht0, image, pose, 1)
    assert set(e1_fields) == {2, 3}
    assert torch.equal(e1_fields[2], teacher)
    assert torch.equal(e1_fields[3], teacher)
    assert len(e1_output) == 5
    assert float(e1_output[4]['tapf_stats'][
        'hierarchical_stage_count']) == 2.0

    e11_fields, e11_output = capture_fields(ht0, image, pose, 11)
    assert set(e11_fields) == {2, 3}
    assert not torch.equal(e11_fields[2], teacher)
    assert not torch.equal(e11_fields[3], teacher)
    assert not torch.equal(e11_fields[2], e11_fields[3])
    stats = e11_output[4]['tapf_stats']
    assert float(stats['stage1_refinement_active']) == 0.0
    assert float(stats['stage2_refinement_active']) == 1.0
    assert float(stats['stage2_posterior_refinement_l1']) > 0.0


def test_gradient_ownership():
    _, ht0, _ = build('ht0')
    image, pose = synthetic_batch()
    labels = torch.tensor([0, 1])
    ht0.train()
    ht0.set_tapf_epoch(11)
    score, feature, _, _, data = ht0(
        image, label=labels, pose_dict=pose)
    pose_loss = data['tapf_pose_loss']
    reid_loss = score.square().mean() + feature.square().mean()
    tapf_parameters = list(ht0.tapf.parameters())
    psg_parameters = list(ht0.psg_modules_dict.parameters())
    backbone_parameters = [ht0.base.conv1.weight]
    assert tensor_grad_norm(pose_loss, tapf_parameters) > 0.0
    assert tensor_grad_norm(pose_loss, psg_parameters) == 0.0
    assert tensor_grad_norm(pose_loss, backbone_parameters) == 0.0
    assert tensor_grad_norm(reid_loss, tapf_parameters) == 0.0
    assert tensor_grad_norm(reid_loss, psg_parameters) > 0.0
    assert tensor_grad_norm(reid_loss, backbone_parameters) > 0.0


def test_eval_external_pose_exact_parity():
    _, ht0, _ = build('ht0')
    image, pose = synthetic_batch()
    shuffled = {
        key: (value.flip(0) if isinstance(value, torch.Tensor) else value)
        for key, value in pose.items()
    }
    ht0.eval()
    with torch.no_grad():
        descriptors = {
            'correct': ht0(image, pose_dict=pose)[0],
            'shuffle': ht0(image, pose_dict=shuffled)[0],
            'none': ht0(image, pose_dict=None)[0],
            'exploding': ht0(image, pose_dict=ExplodingPoseDict())[0],
        }
    expected = descriptors['correct']
    for name, observed in descriptors.items():
        assert bool(torch.isfinite(observed).all()), name
        assert torch.equal(observed, expected), name


def test_strict_reload_and_finite_state():
    _, ht0, _ = build('ht0')
    state = ht0.state_dict()
    assert state
    assert all(bool(torch.isfinite(value.float()).all())
               for value in state.values())
    _, replica, _ = build('ht0')
    result = replica.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    for key, value in state.items():
        assert torch.equal(value, replica.state_dict()[key]), key


def main():
    tests = [
        test_config_contract,
        test_torch_load_compatibility,
        test_b0_existing_path_exact,
        test_matched_initialization_and_topology,
        test_e1_e11_progressive_routes,
        test_gradient_ownership,
        test_eval_external_pose_exact_parity,
        test_strict_reload_and_finite_state,
    ]
    for test in tests:
        test()
        print('PASS ' + test.__name__)
    print('EXP380_RESNET_TAPF_UNIT_PASS %d/%d' % (len(tests), len(tests)))


if __name__ == '__main__':
    main()
