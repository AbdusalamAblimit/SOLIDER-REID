"""CUDA AMP/GradScaler/optimizer/peak-memory preflight for exp376.

Run on each training host with EXP376_SMOKE_BATCH=64 before formal launch.
"""

import os
from pathlib import Path
import sys

import pytest
import torch
from torch.cuda import amp


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import cfg as base_cfg
from model import make_model


CONFIG = ROOT / os.environ.get(
    'EXP376_SMOKE_CONFIG',
    'configs/occluded_duke/exp376_p0_hyper_lora.yml')


def _build():
    cfg = base_cfg.clone()
    cfg.merge_from_file(str(CONFIG))
    cfg.defrost()
    cfg.MODEL.PRETRAIN_PATH = ''
    cfg.MODEL.PRETRAIN_CHOICE = 'none'
    cfg.MODEL.WITH_CP = False
    cfg.freeze()
    model = make_model(
        cfg, num_class=2, camera_num=0, view_num=0,
        semantic_weight=float(cfg.MODEL.SEMANTIC_WEIGHT))
    return model.cuda(), cfg


def _pose_dict(batch):
    heatmaps = torch.zeros(batch, 2, 17, 96, 32, device='cuda')
    y_grid = torch.arange(96, device='cuda').view(1, 96, 1)
    x_grid = torch.arange(32, device='cuda').view(1, 1, 32)
    for joint in range(17):
        cy = 5.0 + joint * 5.2
        cx = 8.0 + (joint % 5) * 4.0
        gaussian = torch.exp(
            -((y_grid - cy).square() / (2 * 4.8 ** 2)
              + (x_grid - cx).square() / (2 * 3.2 ** 2)))
        confidence = 0.55 + 0.45 * joint / 16.0
        heatmaps[:, 0, joint] = confidence * gaussian
    heatmaps[:, 1, :, 70:90, 1:10] = 0.75
    return {
        'heatmaps': heatmaps,
        'scores': torch.ones(batch, 2, 17, device='cuda'),
        'person_mask': torch.ones(batch, 2, device='cuda'),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA preflight')
def test_exp376_production_amp_backward_update_and_peak_memory():
    torch.manual_seed(376020)
    torch.cuda.manual_seed_all(376020)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, cfg = _build()
    model.train()
    batch = int(os.environ.get('EXP376_SMOKE_BATCH', '2'))
    assert batch > 1

    modules = model.pose_hyper_lora_modules
    assert len(modules) == 8
    applied = {}
    handles = []

    def _hook(key):
        def capture(_module, inputs, output):
            before = inputs[0].detach()
            after = output[0].detach()
            diff = after.float() - before.float()
            applied[key] = {
                'changed_fraction': float((after != before).float().mean()),
                'applied_rms': float(diff.square().mean().sqrt()),
            }
        return capture

    for key, module in modules.items():
        handles.append(module.register_forward_hook(_hook(key)))

    parameters = [parameter for module in modules.values()
                  for parameter in module.parameters()]
    optimizer = torch.optim.SGD(
        parameters, lr=float(cfg.SOLVER.BASE_LR),
        weight_decay=float(cfg.SOLVER.WEIGHT_DECAY))
    scaler = amp.GradScaler(init_scale=float(cfg.SOLVER.AMP_INIT_SCALE))
    before_step = {
        name: value.detach().clone()
        for name, value in modules.named_parameters()
    }

    rgb = torch.randn(batch, 3, 384, 128, device='cuda')
    labels = torch.arange(batch, device='cuda') % 2
    pose = _pose_dict(batch)
    optimizer.zero_grad(set_to_none=True)
    try:
        with amp.autocast(enabled=True):
            score, feat, _, _ = model(rgb, label=labels, pose_dict=pose)
            loss = feat.float().square().mean() + score.float().square().mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        for key, module in modules.items():
            for name, parameter in (
                    ('a_basis', module.a_basis),
                    ('b_basis', module.b_basis),
                    ('pose_in', module.pose_mlp[0].weight),
                    ('pose_out', module.pose_mlp[-1].weight),
                    ('alpha', module.residual_scale)):
                assert parameter.grad is not None, (key, name)
                assert torch.isfinite(parameter.grad).all(), (key, name)
                assert parameter.grad.abs().sum() > 0, (key, name)

        scaler.step(optimizer)
        scaler.update()
    finally:
        for handle in handles:
            handle.remove()

    assert set(applied) == set(modules)
    assert min(value['changed_fraction'] for value in applied.values()) > 0.01
    assert min(value['applied_rms'] for value in applied.values()) > 0.0

    changed_groups = {'a_basis': 0, 'b_basis': 0, 'pose_mlp': 0,
                      'residual_scale': 0}
    for name, parameter in modules.named_parameters():
        if torch.equal(before_step[name], parameter.detach()):
            continue
        if name.endswith('a_basis'):
            changed_groups['a_basis'] += 1
        elif name.endswith('b_basis'):
            changed_groups['b_basis'] += 1
        elif '.pose_mlp.' in name:
            changed_groups['pose_mlp'] += 1
        elif name.endswith('residual_scale'):
            changed_groups['residual_scale'] += 1
    assert all(count > 0 for count in changed_groups.values()), changed_groups

    peak_gib = torch.cuda.max_memory_allocated() / 1024 ** 3
    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    assert peak_gib < total_gib
    print({
        'batch': batch,
        'loss': float(loss.detach()),
        'applied': applied,
        'changed_groups': changed_groups,
        'peak_memory_gib': peak_gib,
        'total_memory_gib': total_gib,
    })
