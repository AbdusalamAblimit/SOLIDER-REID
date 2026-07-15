"""Mechanism tests for exp376 Pose Hyper-LoRA."""

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'model' / 'modules' / 'pose_hyper_lora.py'
)
SPEC = importlib.util.spec_from_file_location('pose_hyper_lora', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PoseHyperLoRA = MODULE.PoseHyperLoRA


def _heatmaps(batch, height=32, width=16):
    heatmaps = torch.zeros(batch, 17, height, width)
    for sample in range(batch):
        for joint in range(17):
            y = (2 + joint * 2 + sample) % height
            x = (3 + joint + sample) % width
            heatmaps[sample, joint,
                     max(0, y - 1):min(height, y + 2),
                     max(0, x - 1):min(width, x + 2)] = (
                         0.45 + 0.5 * joint / 16.0)
    return heatmaps


def test_dynamic_transform_identity_pose_dependence_and_gradients():
    torch.manual_seed(376001)
    batch, height, width, channels = 3, 8, 4, 32
    module = PoseHyperLoRA(
        feat_dim=channels, rank=4, num_bases=4,
        pose_hidden_dim=16, residual_scale_init=1e-3)
    x = torch.randn(
        batch, height * width, channels, requires_grad=True)
    heatmaps = _heatmaps(batch)

    output, stats = module(x, (height, width), heatmaps)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert stats['visibility_mean'].item() > 0
    assert stats['coefficient_abs_mean'].item() > 0
    assert stats['delta_rms'].item() > 0

    output.float().square().mean().backward()
    for parameter in (
            module.a_basis, module.b_basis,
            module.pose_mlp[0].weight, module.pose_mlp[-1].weight,
            module.norm.weight, module.residual_scale):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert x.grad is not None and torch.isfinite(x.grad).all()

    module.eval()
    detached = x.detach()
    with torch.no_grad():
        correct, _ = module(detached, (height, width), heatmaps)
        shuffled, _ = module(
            detached, (height, width), heatmaps.roll(1, dims=0))
        correct_visibility_shuffled_coeff, _ = module(
            detached, (height, width), heatmaps.roll(1, dims=0),
            visibility_heatmaps=heatmaps)
        zero, zero_stats = module(
            detached, (height, width), torch.zeros_like(heatmaps))
        absent, absent_stats = module(
            detached, (height, width), None)
    assert not torch.equal(correct, shuffled)
    assert not torch.equal(correct, correct_visibility_shuffled_coeff)
    assert torch.equal(zero, detached)
    assert torch.equal(absent, detached)
    assert zero_stats['visibility_mean'].item() == 0.0
    assert absent_stats['visibility_mean'].item() == 0.0


def test_coefficients_generate_both_sides_of_effective_matrix():
    torch.manual_seed(376002)
    module = PoseHyperLoRA(
        feat_dim=12, rank=3, num_bases=2, pose_hidden_dim=8,
        residual_scale_init=1.0)
    module.eval()
    x = torch.randn(1, 4, 12)
    pose_a = torch.zeros(1, 17, 2, 2)
    pose_b = torch.zeros_like(pose_a)
    pose_a[:, 0] = 1.0
    pose_b[:, 8] = 1.0
    with torch.no_grad():
        coeff_a = torch.tanh(module.pose_mlp(
            pose_a.permute(0, 2, 3, 1).reshape(1, 4, 17)))
        coeff_b = torch.tanh(module.pose_mlp(
            pose_b.permute(0, 2, 3, 1).reshape(1, 4, 17)))
        out_a, _ = module(x, (2, 2), pose_a)
        out_b, _ = module(x, (2, 2), pose_b)
    a_a, b_a = coeff_a.chunk(2, dim=-1)
    a_b, b_b = coeff_b.chunk(2, dim=-1)
    assert not torch.equal(a_a, a_b)
    assert not torch.equal(b_a, b_b)
    assert not torch.equal(out_a, out_b)


def test_cpu_autocast_is_finite():
    torch.manual_seed(376003)
    module = PoseHyperLoRA(feat_dim=16, rank=2, num_bases=2)
    x = torch.randn(2, 8, 16, requires_grad=True)
    pose = _heatmaps(2, 8, 4)
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        output, stats = module(x, (4, 2), pose)
        loss = output.float().square().mean()
    loss.backward()
    assert torch.isfinite(output.float()).all()
    assert all(torch.isfinite(value).all() for value in stats.values())
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in (
            module.a_basis, module.b_basis,
            module.pose_mlp[0].weight, module.pose_mlp[-1].weight,
            module.residual_scale):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_diagonal_control_projection_count_and_pose_dependence():
    torch.manual_seed(376004)
    basis = PoseHyperLoRA(
        feat_dim=32, rank=4, num_bases=4, factorization='basis')
    diagonal = PoseHyperLoRA(
        feat_dim=32, rank=4, num_bases=4, factorization='diagonal')
    basis_projection_params = basis.a_basis.numel() + basis.b_basis.numel()
    diagonal_projection_params = (
        diagonal.a_basis.numel() + diagonal.b_basis.numel())
    assert basis_projection_params == diagonal_projection_params

    x = torch.randn(2, 8, 32)
    pose = _heatmaps(2, 8, 4)
    with torch.no_grad():
        correct, stats = diagonal(x, (4, 2), pose)
        shuffled, _ = diagonal(x, (4, 2), pose.roll(1, dims=0))
        zero, _ = diagonal(x, (4, 2), torch.zeros_like(pose))
    assert stats['delta_rms'].item() > 0
    assert not torch.equal(correct, shuffled)
    assert torch.equal(zero, x)


def test_heatmap_batch_mismatch_fails_loudly():
    module = PoseHyperLoRA(feat_dim=8)
    x = torch.randn(2, 4, 8)
    pose = torch.zeros(1, 17, 2, 2)
    try:
        module(x, (2, 2), pose)
    except ValueError as error:
        assert 'batch size' in str(error)
    else:
        raise AssertionError('batch-mismatched pose must not broadcast')
