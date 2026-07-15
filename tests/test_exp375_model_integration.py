"""Real-model integration smoke for exp375 B0/M0/P0 configs.

Run this test separately on a CUDA worker before formal training.  It builds
the production PoseBackboneModel, but disables checkpoint I/O so the smoke
tests code/config/data-flow rather than external assets.
"""

from pathlib import Path

import torch

from config import cfg as base_cfg
from model import make_model
from model.pose_backbone_model import PoseBackboneModel


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    'b0': ROOT / 'configs/occluded_duke/exp375_b0.yml',
    'm0': ROOT / 'configs/occluded_duke/exp375_m0_canonical.yml',
    'p0': ROOT / 'configs/occluded_duke/exp375_p0_prsm.yml',
}


def _resolved_config(path):
    cfg = base_cfg.clone()
    cfg.merge_from_file(str(path))
    cfg.defrost()
    cfg.MODEL.PRETRAIN_PATH = ''
    cfg.MODEL.PRETRAIN_CHOICE = 'none'
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.WITH_CP = False
    cfg.freeze()
    return cfg


def _build(path, device):
    cfg = _resolved_config(path)
    model = make_model(
        cfg, num_class=2, camera_num=0, view_num=0,
        semantic_weight=float(cfg.MODEL.SEMANTIC_WEIGHT))
    assert type(model) is PoseBackboneModel
    model.to(device)
    model.eval()
    return model, cfg


def _pose_dict(batch, device):
    heatmaps = torch.zeros(batch, 2, 17, 96, 32, device=device)
    ys = torch.tensor(
        [5, 5, 5, 6, 6, 18, 18, 30, 30, 42, 42, 48, 48,
         66, 66, 89, 89], device=device)
    xs = torch.tensor(
        [16, 14, 18, 12, 20, 11, 21, 8, 24, 6, 26, 13, 19,
         13, 19, 13, 19], device=device)
    for sample in range(batch):
        for joint in range(17):
            y = int((ys[joint] + sample * 2).clamp(max=95))
            x = int((xs[joint] + sample).clamp(max=31))
            y0, y1 = max(0, y - 3), min(96, y + 4)
            x0, x1 = max(0, x - 3), min(32, x + 4)
            heatmaps[sample, 0, joint, y0:y1, x0:x1] = 1.0
    # A valid, spatially distinct distractor makes scene-merged pose differ
    # from the target.  The integration hook below proves PRSM sees person 0.
    heatmaps[:, 1, :, 76:88, 2:10] = 0.8
    scores = torch.ones(batch, 2, 17, device=device)
    person_mask = torch.ones(batch, 2, device=device)
    return {
        'heatmaps': heatmaps,
        'scores': scores,
        'person_mask': person_mask,
    }


def _assert_eval_output(output, batch):
    assert isinstance(output, tuple) and len(output) == 2
    descriptor, feature_maps = output
    assert descriptor.shape == (batch, 768)
    assert torch.isfinite(descriptor).all()
    assert len(feature_maps) == 4
    assert feature_maps[-1].shape == (batch, 768, 12, 4)
    assert torch.isfinite(feature_maps[-1]).all()
    return descriptor, feature_maps


def test_exp375_real_model_config_forward_backward_and_reload():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = 2
    torch.manual_seed(375010)
    b0, b0_cfg = _build(CONFIGS['b0'], device)
    torch.manual_seed(375010)
    m0, m0_cfg = _build(CONFIGS['m0'], device)
    torch.manual_seed(375010)
    p0, p0_cfg = _build(CONFIGS['p0'], device)

    assert not b0.use_prsm
    assert m0.use_prsm and p0.use_prsm
    assert m0.prsm_pose_source == 'canonical'
    assert p0.prsm_pose_source == 'input'
    assert b0.use_target_heatmap
    assert m0.use_target_heatmap
    assert p0.use_target_heatmap
    assert b0.psg_stage_indices == set()
    assert m0.psg_stage_indices == set()
    assert p0.psg_stage_indices == set()
    assert m0_cfg.SOLVER.IMS_PER_BATCH == p0_cfg.SOLVER.IMS_PER_BATCH == 64
    assert m0_cfg.SOLVER.SEED == p0_cfg.SOLVER.SEED == 1234

    m0_state = m0.state_dict()
    p0_state = p0.state_dict()
    assert set(m0_state) == set(p0_state)
    for key in m0_state:
        assert torch.equal(m0_state[key], p0_state[key]), key

    rgb = torch.randn(batch, 3, 384, 128, device=device)
    pose = _pose_dict(batch, device)
    target_heatmaps = pose['heatmaps'][:, 0]
    scene_heatmaps = pose['heatmaps'].amax(dim=1)
    assert not torch.equal(target_heatmaps, scene_heatmaps)
    captured_prsm_heatmaps = []

    def capture_prsm_input(_module, inputs):
        assert len(inputs) == 2
        captured_prsm_heatmaps.append(inputs[1].detach().clone())

    capture_handle = p0.prsm.register_forward_pre_hook(capture_prsm_input)
    with torch.no_grad():
        b0_descriptor, _ = _assert_eval_output(
            b0(rgb, pose_dict=pose), batch)
        m0_descriptor, _ = _assert_eval_output(
            m0(rgb, pose_dict=pose), batch)
    try:
        p0_descriptor, _ = _assert_eval_output(
            p0(rgb, pose_dict=pose), batch)
    finally:
        capture_handle.remove()
    assert len(captured_prsm_heatmaps) == 1
    assert torch.equal(captured_prsm_heatmaps[0], target_heatmaps)
    assert not torch.equal(captured_prsm_heatmaps[0], scene_heatmaps)
    assert not torch.equal(p0_descriptor.detach(), m0_descriptor)
    assert not torch.equal(p0_descriptor.detach(), b0_descriptor)

    p0_descriptor.float().square().mean().backward()
    for parameter in (
            p0.prsm.candidate_proj.weight,
            p0.prsm.read_query_proj.weight,
            p0.prsm.part_keys,
            p0.prsm.retention_logits,
            p0.prsm.output_proj.weight,
            p0.prsm.residual_scale):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0

    frozen_state = {
        key: value.detach().cpu().clone()
        for key, value in p0.state_dict().items()
    }
    del b0, m0
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    torch.manual_seed(999)
    reloaded, _ = _build(CONFIGS['p0'], device)
    incompatible = reloaded.load_state_dict(frozen_state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    with torch.no_grad():
        reloaded_descriptor, _ = _assert_eval_output(
            reloaded(rgb, pose_dict=pose), batch)
    assert torch.equal(p0_descriptor.detach(), reloaded_descriptor)


if __name__ == '__main__':
    test_exp375_real_model_config_forward_backward_and_reload()
    print('exp375 real-model integration smoke: PASS')
