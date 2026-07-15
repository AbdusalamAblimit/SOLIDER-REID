"""Production-model integration smoke for exp376 P0/M0 configs."""

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import cfg as base_cfg
from model import make_model
from model.pose_backbone_model import PoseBackboneModel


P0_CONFIG = ROOT / 'configs/occluded_duke/exp376_p0_hyper_lora.yml'
M0_CONFIG = (
    ROOT / 'configs/occluded_duke/exp376_m0_canonical_hyper_lora.yml')


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
    model.to(device).eval()
    return model, cfg


def _pose_dict(batch, device):
    heatmaps = torch.zeros(batch, 2, 17, 96, 32, device=device)
    for sample in range(batch):
        for joint in range(17):
            y = min(94, 4 + joint * 5 + sample)
            x = 5 + (joint * 3 + sample) % 22
            heatmaps[sample, 0, joint, y:y + 2, x:x + 2] = (
                0.5 + 0.5 * joint / 16.0)
    heatmaps[:, 1, :, 76:88, 2:10] = 0.8
    return {
        'heatmaps': heatmaps,
        'scores': torch.ones(batch, 2, 17, device=device),
        'person_mask': torch.ones(batch, 2, device=device),
    }


def test_exp376_configs_pair_initialization_target_pose_and_reload():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(376010)
    p0, p0_cfg = _build(P0_CONFIG, device)
    torch.manual_seed(376010)
    m0, m0_cfg = _build(M0_CONFIG, device)

    assert p0.use_pose_hyper_lora and m0.use_pose_hyper_lora
    assert p0.pose_hyper_lora_stage_indices == {2, 3}
    assert m0.pose_hyper_lora_stage_indices == {2, 3}
    assert len(p0.pose_hyper_lora_modules) == 8
    assert len(m0.pose_hyper_lora_modules) == 8
    assert p0.pose_hyper_lora_pose_source == 'input'
    assert m0.pose_hyper_lora_pose_source == 'canonical'
    assert p0.psg_stage_indices == m0.psg_stage_indices == set()
    assert not p0.use_prsm and not m0.use_prsm
    assert p0_cfg.SOLVER.IMS_PER_BATCH == m0_cfg.SOLVER.IMS_PER_BATCH == 64

    p0_state = p0.state_dict()
    m0_state = m0.state_dict()
    assert set(p0_state) == set(m0_state)
    for key in p0_state:
        assert torch.equal(p0_state[key], m0_state[key]), key

    batch = 1
    rgb = torch.randn(batch, 3, 384, 128, device=device)
    pose = _pose_dict(batch, device)
    target = pose['heatmaps'][:, 0]
    scene = pose['heatmaps'].amax(dim=1)
    assert not torch.equal(target, scene)

    captured = []
    first_module = p0.pose_hyper_lora_modules['s2_b0']

    def _capture(_module, inputs):
        captured.append(inputs[2].detach().clone())

    handle = first_module.register_forward_pre_hook(_capture)
    try:
        with torch.no_grad():
            p0_descriptor, p0_maps = p0(rgb, pose_dict=pose)
            m0_descriptor, m0_maps = m0(rgb, pose_dict=pose)
    finally:
        handle.remove()

    assert len(captured) == 1
    assert torch.equal(captured[0], target)
    assert not torch.equal(captured[0], scene)
    assert p0_descriptor.shape == m0_descriptor.shape == (batch, 768)
    assert p0_maps[-1].shape == m0_maps[-1].shape == (batch, 768, 12, 4)
    assert torch.isfinite(p0_descriptor).all()
    assert torch.isfinite(m0_descriptor).all()
    assert not torch.equal(p0_descriptor, m0_descriptor)
    assert set(p0._last_pose_hyper_lora_stats) == {
        's2_b0', 's2_b1', 's2_b2', 's2_b3', 's2_b4', 's2_b5',
        's3_b0', 's3_b1'}

    frozen = {key: value.detach().cpu().clone()
              for key, value in p0.state_dict().items()}
    del m0
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    torch.manual_seed(999)
    reloaded, _ = _build(P0_CONFIG, device)
    incompatible = reloaded.load_state_dict(frozen, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    with torch.no_grad():
        reloaded_descriptor, _ = reloaded(rgb, pose_dict=pose)
    assert torch.equal(p0_descriptor, reloaded_descriptor)


if __name__ == '__main__':
    test_exp376_configs_pair_initialization_target_pose_and_reload()
    print('exp376 production-model integration smoke: PASS')
