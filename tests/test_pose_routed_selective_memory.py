"""Mechanism checks for exp375 Pose-Routed Selective Memory."""

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'model' / 'modules' / 'pose_routed_selective_memory.py'
)
SPEC = importlib.util.spec_from_file_location(
    'pose_routed_selective_memory', MODULE_PATH)
PRSM_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PRSM_MODULE)
PoseRoutedSelectiveMemory = PRSM_MODULE.PoseRoutedSelectiveMemory


def make_heatmaps(batch, height=32, width=16):
    heatmaps = torch.zeros(batch, 17, height, width)
    base_y = torch.tensor([
        2, 2, 2, 3, 3, 7, 7, 11, 11, 15, 15, 16, 16, 22, 22, 29, 29])
    base_x = torch.tensor([
        8, 7, 9, 6, 10, 6, 10, 4, 12, 3, 13, 7, 9, 7, 9, 7, 9])
    for sample in range(batch):
        for joint in range(17):
            y = int((base_y[joint] + sample).clamp(max=height - 1))
            x = int((base_x[joint] + sample).clamp(max=width - 1))
            y0, y1 = max(0, y - 2), min(height, y + 3)
            x0, x1 = max(0, x - 2), min(width, x + 3)
            heatmaps[sample, joint, y0:y1, x0:x1] = 1.0
    return heatmaps


def main():
    torch.manual_seed(375001)
    batch, channels, height, width = 3, 32, 8, 4
    features = torch.randn(
        batch, channels, height, width, requires_grad=True)
    heatmaps = make_heatmaps(batch)

    module = PoseRoutedSelectiveMemory(
        feat_dim=channels, state_dim=16, routing='parts',
        residual_scale_init=1e-3, bidirectional=True)
    module.train()
    output, stats = module(features, heatmaps)
    assert output.shape == features.shape
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(value).all() for value in stats.values())
    output.square().mean().backward()
    for parameter in (
            module.candidate_proj.weight, module.read_query_proj.weight,
            module.part_keys, module.retention_logits,
            module.output_proj.weight, module.residual_scale):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert features.grad is not None and torch.isfinite(features.grad).all()

    # A true zero pose writes no state and is an exact identity path.
    module.eval()
    with torch.no_grad():
        zero_output, zero_stats = module(
            features.detach(), torch.zeros_like(heatmaps))
    assert torch.equal(zero_output, features.detach())
    assert zero_stats['prsm_write_mean'].item() == 0.0

    # Instance pose changes recurrent writes for the same RGB feature map.
    with torch.no_grad():
        correct, _ = module(features.detach(), heatmaps)
        shuffled, _ = module(features.detach(), heatmaps.roll(1, dims=0))
    assert not torch.equal(correct, shuffled)

    # Shared forward/reverse parameters make the bidirectional module
    # equivariant to a simultaneous vertical flip of RGB and pose.
    with torch.no_grad():
        flipped, _ = module(
            features.detach().flip(2), heatmaps.flip(2))
    assert torch.allclose(
        flipped, correct.flip(2), atol=2e-6, rtol=2e-5)

    # The uniform parameter-matched control ignores any supplied heatmap.
    uniform = PoseRoutedSelectiveMemory(
        feat_dim=channels, state_dim=16, routing='uniform',
        residual_scale_init=1e-3, bidirectional=True)
    uniform.eval()
    with torch.no_grad():
        uniform_a, _ = uniform(features.detach(), heatmaps)
        uniform_b, _ = uniform(features.detach(), heatmaps.roll(1, dims=0))
        uniform_none, _ = uniform(features.detach(), None)
    assert torch.equal(uniform_a, uniform_b)
    assert torch.equal(uniform_a, uniform_none)

    # CPU autocast smoke for the short recurrent loop.
    amp_module = PoseRoutedSelectiveMemory(
        feat_dim=channels, state_dim=16, routing='parts')
    amp_features = features.detach().clone().requires_grad_(True)
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        amp_output, amp_stats = amp_module(amp_features, heatmaps)
        amp_loss = amp_output.float().square().mean()
    amp_loss.backward()
    assert torch.isfinite(amp_output.float()).all()
    assert all(torch.isfinite(value).all() for value in amp_stats.values())
    assert amp_features.grad is not None
    assert torch.isfinite(amp_features.grad).all()

    print('PRSM mechanism checks: PASS')


if __name__ == '__main__':
    main()
