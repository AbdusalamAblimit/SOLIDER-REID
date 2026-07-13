"""Deterministic mechanism checks for PBSR.

Run with the repository-local uv environment:
    .venv/bin/python tests/test_pose_bidirectional_router.py
"""

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'model' / 'modules' / 'pose_bidirectional_router.py'
)
SPEC = importlib.util.spec_from_file_location('pose_bidirectional_router', MODULE_PATH)
PBSR_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PBSR_MODULE)
PoseSupervisedBidirectionalRouter = (
    PBSR_MODULE.PoseSupervisedBidirectionalRouter
)


def make_heatmaps(batch, height=32, width=16):
    heatmaps = torch.zeros(batch, 17, height, width)
    ys = torch.linspace(2, height - 3, 17).round().long()
    xs = torch.tensor([(i % 4) + width // 3 for i in range(17)])
    for b in range(batch):
        for keypoint in range(17):
            y = int((ys[keypoint] + b).clamp(max=height - 1))
            x = int((xs[keypoint] + b).clamp(max=width - 1))
            heatmaps[b, keypoint, y, x] = 1.0
    return heatmaps


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def clear_grads(module, tensor=None):
    module.zero_grad(set_to_none=True)
    if tensor is not None:
        tensor.grad = None


def main():
    torch.manual_seed(20260713)
    batch, channels, height, width = 3, 64, 8, 4
    features = torch.randn(
        batch, channels, height, width, requires_grad=True)
    heatmaps = make_heatmaps(batch)

    router = PoseSupervisedBidirectionalRouter(
        feat_dim=channels,
        route_dim=32,
        num_slots=6,
        num_heads=4,
        slot_mixer=True,
        writeback=True,
        coupled_write=True,
        supervision='correct',
    )
    router.train()

    # Zero-gated PBSR is an exact baseline, not merely numerically close.
    refined, aux = router(features, heatmaps)
    assert torch.equal(refined, features)
    assert refined.shape == features.shape
    assert torch.isfinite(refined).all()
    assert torch.isfinite(aux['pbsr_route_loss'])
    routes = aux['pbsr_routes']
    assert routes.shape == (batch, 7, height * width)
    assert torch.allclose(
        routes.sum(dim=-1), torch.ones(batch, 7), atol=1e-6)

    # Route supervision updates the router but cannot reach backbone/input.
    aux['pbsr_route_loss'].backward()
    assert features.grad is None or torch.count_nonzero(features.grad) == 0
    key_grad = router.key_proj.weight.grad
    query_grad = router.slot_queries.grad
    assert key_grad is not None and torch.isfinite(key_grad).all()
    assert query_grad is not None and torch.isfinite(query_grad).all()
    assert key_grad.abs().sum() > 0
    assert query_grad.abs().sum() > 0

    # At alpha=0, identity loss must first receive a finite non-zero gate grad.
    clear_grads(router, features)
    refined, _ = router(features, heatmaps)
    identity_probe = refined.square().mean()
    identity_probe.backward()
    assert router.write_scale.grad is not None
    assert torch.isfinite(router.write_scale.grad)
    assert router.write_scale.grad.abs() > 0
    assert features.grad is not None and torch.isfinite(features.grad).all()

    # Once the gate is open, identity loss reaches write path and router.
    clear_grads(router, features)
    with torch.no_grad():
        router.write_scale.fill_(0.1)
    refined, _ = router(features, heatmaps)
    refined.square().mean().backward()
    assert router.out_proj.weight.grad is not None
    assert router.key_proj.weight.grad is not None
    assert router.slot_queries.grad is not None
    assert router.out_proj.weight.grad.abs().sum() > 0
    assert router.key_proj.weight.grad.abs().sum() > 0
    assert router.slot_queries.grad.abs().sum() > 0

    # Coupled and independent-write controls have identical parameters.
    independent = PoseSupervisedBidirectionalRouter(
        feat_dim=channels,
        route_dim=32,
        num_slots=6,
        num_heads=4,
        slot_mixer=True,
        writeback=True,
        coupled_write=False,
        supervision='correct',
    )
    assert parameter_count(router) == parameter_count(independent)

    # The representation path is pose-free: eval output is bitwise invariant
    # to absent, correct, or random heatmaps.
    router.eval()
    with torch.no_grad():
        output_none, _ = router(features.detach(), None)
        output_correct, _ = router(features.detach(), heatmaps)
        output_random, _ = router(
            features.detach(), torch.randn_like(heatmaps))
    assert torch.equal(output_none, output_correct)
    assert torch.equal(output_none, output_random)

    # Shuffled supervision is deterministic and does not consume global RNG.
    shuffled = PoseSupervisedBidirectionalRouter(
        feat_dim=channels,
        route_dim=32,
        num_slots=6,
        num_heads=4,
        supervision='shuffled',
    )
    shuffled.train()
    rng_before = torch.get_rng_state().clone()
    shuffled(features.detach(), heatmaps)
    rng_after = torch.get_rng_state()
    assert torch.equal(rng_before, rng_after)

    # CPU autocast smoke: finite forward and backward.
    amp_router = PoseSupervisedBidirectionalRouter(
        feat_dim=channels,
        route_dim=32,
        num_slots=6,
        num_heads=4,
        supervision='correct',
    )
    amp_router.train()
    amp_features = features.detach().clone().requires_grad_(True)
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        amp_output, amp_aux = amp_router(amp_features, heatmaps)
        amp_loss = amp_output.float().square().mean() + amp_aux['pbsr_route_loss']
    amp_loss.backward()
    assert torch.isfinite(amp_output.float()).all()
    assert torch.isfinite(amp_loss)
    assert amp_features.grad is not None
    assert torch.isfinite(amp_features.grad).all()

    print('PBSR mechanism checks: PASS')


if __name__ == '__main__':
    main()
