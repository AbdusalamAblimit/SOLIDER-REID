"""CPU unit and model-invariant gates for exp381 ViT TAPF."""
import random
import sys
import types
import importlib
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# The local uv environment intentionally lacks two optional Swin-only import
# dependencies.  exp381 does not execute Swin code; production CUDA gates run
# in the complete 4090 environment.  Minimal import stubs keep this CPU unit
# isolated without pretending to validate either dependency.
try:
    import cv2  # noqa: F401
except ImportError:
    sys.modules['cv2'] = types.ModuleType('cv2')
try:
    import mmcv.runner  # noqa: F401
except ImportError:
    mmengine = types.ModuleType('mmengine')
    runner = types.ModuleType('mmengine.runner')

    def unavailable_load_checkpoint(*_args, **_kwargs):
        raise RuntimeError('Swin checkpoint loading is unavailable in CPU unit')

    runner.load_checkpoint = unavailable_load_checkpoint
    mmengine.runner = runner
    sys.modules['mmengine'] = mmengine
    sys.modules['mmengine.runner'] = runner

from config import cfg as root_cfg
from model.backbones.vit_pytorch import TransReID, torch_load_compat
from model.vit_tapf_model import VitTapfModel


CONFIGS = {
    'b0': 'configs/occluded_duke/exp381_vit_b0.yml',
    'd0': 'configs/occluded_duke/exp381_vit_d0.yml',
    'ht0': 'configs/occluded_duke/exp381_vit_ht0.yml',
}
FACTORY_KEY = 'vit_base_patch16_224_TransReID'


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


def tiny_vit_factory(**kwargs):
    model = TransReID(
        img_size=kwargs['img_size'],
        patch_size=8,
        stride_size=kwargs['stride_size'],
        embed_dim=32,
        depth=12,
        num_heads=4,
        mlp_ratio=2,
        qkv_bias=True,
        drop_path_rate=kwargs['drop_path_rate'],
        drop_rate=kwargs['drop_rate'],
        attn_drop_rate=kwargs['attn_drop_rate'],
        camera=kwargs['camera'],
        view=kwargs['view'],
        local_feature=kwargs['local_feature'],
        sie_xishu=kwargs['sie_xishu'],
        gem_pool=kwargs['gem_pool'],
        stem_conv=kwargs['stem_conv'],
    )
    model.in_planes = 32
    return model


def load_config(name):
    config = root_cfg.clone()
    config.defrost()
    config.merge_from_file(str(ROOT / CONFIGS[name]))
    # Unit gates use a 32-d tiny ViT with the same 12-block topology.  The
    # production gate separately validates the real ViT-B checkpoint.
    config.MODEL.PRETRAIN_CHOICE = 'none'
    config.MODEL.PRETRAIN_PATH = ''
    config.MODEL.STRIDE_SIZE = [8, 8]
    config.MODEL.DROP_PATH = 0.0
    config.MODEL.POSE_PFM_HIDDEN = 8
    config.MODEL.POSE_TAPF_HIDDEN = 8
    config.MODEL.POSE_HEATMAP_SIZE = [16, 8]
    config.INPUT.SIZE_TRAIN = [32, 16]
    config.INPUT.SIZE_TEST = [32, 16]
    config.DATALOADER.NUM_WORKERS = 0
    config.freeze()
    return config


def build(name):
    config = load_config(name)
    set_seed(int(config.SOLVER.SEED))
    model = VitTapfModel(
        num_classes=7,
        camera_num=0,
        view_num=0,
        cfg=config,
        factory={FACTORY_KEY: tiny_vit_factory},
    )
    return config, model, torch.get_rng_state().clone()


def synthetic_batch(batch=2):
    generator = torch.Generator().manual_seed(321)
    image = torch.randn(batch, 3, 32, 16, generator=generator)
    heatmaps = torch.rand(batch, 1, 17, 16, 8, generator=generator)
    scores = torch.rand(batch, 1, 17, generator=generator)
    person_mask = torch.ones(batch, 1)
    return image, {
        'heatmaps': heatmaps,
        'scores': scores,
        'person_mask': person_mask,
    }


def tensor_grad_norm(objective, parameters):
    parameters = [parameter for parameter in parameters
                  if parameter.requires_grad]
    gradients = torch.autograd.grad(
        objective, parameters, retain_graph=True, allow_unused=True)
    total = 0.0
    for gradient in gradients:
        if gradient is not None:
            assert bool(torch.isfinite(gradient).all())
            total += float(gradient.detach().float().square().sum())
    return total ** 0.5


def test_config_contract():
    b0 = load_config('b0')
    d0 = load_config('d0')
    ht0 = load_config('ht0')
    assert b0.MODEL.NAME == d0.MODEL.NAME == ht0.MODEL.NAME == 'transformer'
    assert b0.MODEL.VIT_TAPF_EXPERIMENT
    assert not b0.MODEL.POSE_ENABLED and not b0.MODEL.POSE_TAPF
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
        ('trusted-local-checkpoint.pth', {
            'map_location': 'cpu', 'weights_only': False}),
        ('trusted-local-checkpoint.pth', {'map_location': 'cpu'}),
    ]


def test_matched_initialization_and_topology():
    _, b0, b0_rng = build('b0')
    _, d0, d0_rng = build('d0')
    _, ht0, ht0_rng = build('ht0')
    assert d0.psg_group_indices == {3}
    assert ht0.psg_group_indices == {2, 3}
    assert ht0.tapf.source_stages == (1, 2)
    assert len(ht0.tapf.stage_projections) == 2

    for prefix in ('base.', 'bottleneck.', 'classifier.'):
        keys = [key for key in b0.state_dict() if key.startswith(prefix)]
        assert keys
        for key in keys:
            assert torch.equal(b0.state_dict()[key], d0.state_dict()[key]), key
            assert torch.equal(b0.state_dict()[key], ht0.state_dict()[key]), key
    d0_psg = d0.psg_modules_dict.state_dict()
    ht0_psg = ht0.psg_modules_dict.state_dict()
    g3_keys = [key for key in d0_psg if key.startswith('g3_')]
    assert g3_keys and all(key in ht0_psg for key in g3_keys)
    for key in g3_keys:
        assert torch.equal(d0_psg[key], ht0_psg[key]), key
    assert torch.equal(b0_rng, d0_rng)
    assert torch.equal(b0_rng, ht0_rng)


def test_make_model_routes_only_exp381_flag():
    make_model_module = importlib.import_module('model.make_model')
    original = make_model_module.__factory_T_type[FACTORY_KEY]
    make_model_module.__factory_T_type[FACTORY_KEY] = tiny_vit_factory
    try:
        config = load_config('b0')
        set_seed(int(config.SOLVER.SEED))
        model = make_model_module.make_model(
            config, num_class=7, camera_num=0, view_num=0,
            semantic_weight=config.MODEL.SEMANTIC_WEIGHT)
    finally:
        make_model_module.__factory_T_type[FACTORY_KEY] = original
    assert type(model) is VitTapfModel
    assert not model.use_tapf


def test_cls_exact_spatial_bypass():
    _, d0, _ = build('d0')
    gate = d0.psg_modules_dict['g3_b9']
    with torch.no_grad():
        gate.encoder[-1].weight.zero_()
        gate.encoder[-1].bias.fill_(1.0)
    generator = torch.Generator().manual_seed(99)
    tokens = torch.randn(2, 9, 32, generator=generator)
    field = torch.zeros(2, 17, 16, 8)
    observed = d0._apply_psg_tokens(tokens, 3, 9, field)
    assert torch.equal(observed[:, :1], tokens[:, :1])
    assert torch.equal(observed[:, 1:], tokens[:, 1:] * 2.0)


def capture_fields(model, image, pose, epoch):
    captured = {}
    handles = []
    for group_idx, block_idx in ((2, 6), (3, 9)):
        key = 'g%d_b%d' % (group_idx, block_idx)
        if key not in model.psg_modules_dict:
            continue

        def hook(_module, inputs, group_idx=group_idx):
            captured[group_idx] = inputs[2].detach().clone()

        handles.append(
            model.psg_modules_dict[key].register_forward_pre_hook(hook))
    model.train()
    model.set_tapf_epoch(epoch)
    try:
        with torch.no_grad():
            output = model(
                image,
                label=torch.zeros(image.shape[0], dtype=torch.long),
                pose_dict=pose)
    finally:
        for handle in handles:
            handle.remove()
    return captured, output


def test_b0_and_e1_e11_progressive_routes():
    _, b0, _ = build('b0')
    image, pose = synthetic_batch()
    b0.train()
    b0_output = b0(image)
    assert len(b0_output) == 3 and len(b0_output[2]) == 4

    _, ht0, _ = build('ht0')
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
    backbone_parameters = [ht0.base.patch_embed.proj.weight]
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
    ht0.set_tapf_epoch(11)
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
        test_matched_initialization_and_topology,
        test_make_model_routes_only_exp381_flag,
        test_cls_exact_spatial_bypass,
        test_b0_and_e1_e11_progressive_routes,
        test_gradient_ownership,
        test_eval_external_pose_exact_parity,
        test_strict_reload_and_finite_state,
    ]
    for test in tests:
        test()
        print('PASS ' + test.__name__)
    print('EXP381_VIT_TAPF_UNIT_PASS %d/%d' % (len(tests), len(tests)))


if __name__ == '__main__':
    main()
