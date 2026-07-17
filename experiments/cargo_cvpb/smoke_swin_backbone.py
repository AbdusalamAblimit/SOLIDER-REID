# encoding: utf-8
"""
Smoke test for the Swin-Small backbone option in the AFD/OVLI CARGO trainer.

Verifies (no training, ~a few forward/backward passes):
  1. resnet50 path (default) still builds + forwards with the unchanged contract
     (train dict {global_feat,bn_feat,logits,band_w}; eval -> L2-normed BN feat).
  2. swin_small path builds (optionally loading the SOLIDER teacher checkpoint),
     in_planes == 768, forward gives the same train/eval contract, BN feat is
     unit-normalized at eval.
  3. The OVLI hook fires on model.layer4 for swin and captures a (B,768,H,W) NCHW
     map; OVLIHead projects K=gh*gw tokens of dim ovli_dim with per-token unit
     norm; ovli.loss(...) returns a finite scalar that backprops into the Swin
     backbone (grad reaches a Swin parameter -> hook is NOT detached).
  4. OVP loss (feat_dim = model.in_planes = 768) computes a finite scalar on swin.

Run on a CUDA box that has the SOLIDER swin_small.pth (lab-3090-d / lab-4090):
    cd <repo>/experiments/cargo_cvpb
    python3 smoke_swin_backbone.py --swin_pretrain <repo>/pretrained/swin_small.pth

The Swin forward hard-codes .cuda() for the semantic weight, so this smoke
requires CUDA (it asserts on it).
"""
import os
import sys
import types
import importlib.util
import argparse

import torch
import torch.nn.functional as F

# Import wiring.  cargo_cvpb/afd_train.py does `from afd_train import (...)` MEANING
# the afd_reid trainer (it relies on running as __main__ so the name `afd_train` is
# free).  Importing it as a library here would collide, so we:
#   1) put afd_reid first on sys.path and import the REAL afd_model (Swin model),
#   2) load afd_reid/afd_train UNDER the name `afd_train` into sys.modules,
#   3) load cargo_cvpb/afd_train.py under a distinct name so its internal
#      `from afd_train import ...` resolves to (2) -> no circular self-import.
_HERE = os.path.dirname(os.path.abspath(__file__))
_AFD_REID = os.path.join(_HERE, '..', 'afd_reid')
sys.path.insert(0, _AFD_REID)
sys.path.insert(0, _HERE)

from afd_model import build_model                     # noqa: E402  (real Swin/resnet model)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# (2) the afd_reid trainer registered as `afd_train` (what cargo_cvpb expects)
_load_module('afd_train', os.path.join(_AFD_REID, 'afd_train.py'))
# (3) the cargo_cvpb trainer (OVLIHead / OVPMemory live here) under its own name
_cvpb = _load_module('cvpb_afd_train', os.path.join(_HERE, 'afd_train.py'))
OVLIHead, OVPMemory = _cvpb.OVLIHead, _cvpb.OVPMemory


def _args(backbone, swin_pretrain, img_size):
    """Minimal argparse-like namespace for build_model."""
    ns = argparse.Namespace()
    ns.last_stride = 1
    ns.pool = 'gem'
    ns.use_afd = False
    ns.afd_router = True
    ns.afd_cvfc = True
    ns.afd_stage = 'layer1'
    ns.router_cond_view = True
    ns.low_r, ns.mid_r, ns.high_drop_p = 0.125, 0.30, 0.5
    ns.backbone = backbone
    ns.swin_pretrain = swin_pretrain
    ns.swin_semantic_weight = 0.2
    ns.img_size = img_size
    return ns


def _make_batch(B, img_size, device, n_pids=4):
    """Two-view PK-ish batch: B images, n_pids identities, views in {0,1}."""
    H, W = img_size
    imgs = torch.randn(B, 3, H, W, device=device)
    labels = torch.randint(0, n_pids, (B,), device=device)
    views = torch.randint(0, 2, (B,), device=device)
    # guarantee at least one opposite-view positive+negative so OVLI loss is > 0
    labels[:B // 2] = labels[B // 2:]         # mirror pids across the two halves
    views[:B // 2] = 0
    views[B // 2:] = 1
    return imgs, labels, views


def check_resnet(device):
    print("\n[1] resnet50 path (must stay the unchanged contract)")
    model = build_model(num_classes=10, args=_args('resnet50', '', (256, 128))).to(device)
    assert model.in_planes == 2048, model.in_planes
    assert model.backbone == 'resnet50'
    imgs, labels, views = _make_batch(8, (256, 128), device)
    model.train()
    out = model(imgs, view_idx=None, return_cvfc=False)
    for k in ('global_feat', 'bn_feat', 'logits', 'band_w'):
        assert k in out, f"missing train key {k}"
    assert out['global_feat'].shape == (8, 2048), out['global_feat'].shape
    assert out['logits'].shape == (8, 10), out['logits'].shape
    model.eval()
    with torch.no_grad():
        ef = model(imgs, view_idx=None)
    assert ef.shape == (8, 2048), ef.shape
    norms = ef.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), norms[:4]
    max_dev = (norms - 1).abs().max().item()
    print(f"    OK  in_planes=2048  train logits {tuple(out['logits'].shape)}  "
          f"eval BN unit-norm (max|n-1|={max_dev:.2e})")
    return model.in_planes


def check_swin(device, swin_pretrain, img_size):
    print(f"\n[2] swin_small path  (img_size={img_size}, "
          f"pretrain={'yes' if swin_pretrain else 'NO -> scratch'})")
    model = build_model(num_classes=10,
                        args=_args('swin_small', swin_pretrain, img_size)).to(device)
    assert model.backbone == 'swin_small'
    assert model.in_planes == 768, f"swin in_planes should be 768, got {model.in_planes}"
    assert isinstance(model.layer4, torch.nn.Identity), type(model.layer4)
    imgs, labels, views = _make_batch(8, img_size, device)

    model.train()
    out = model(imgs, view_idx=None, return_cvfc=False)
    for k in ('global_feat', 'bn_feat', 'logits', 'band_w'):
        assert k in out, f"missing train key {k}"
    assert out['global_feat'].shape == (8, 768), out['global_feat'].shape
    assert out['bn_feat'].shape == (8, 768), out['bn_feat'].shape
    assert out['logits'].shape == (8, 10), out['logits'].shape
    assert out['band_w'] is None, "swin has no band weights"
    # the model must produce finite features
    assert torch.isfinite(out['global_feat']).all(), "non-finite swin global_feat"

    model.eval()
    with torch.no_grad():
        ef = model(imgs, view_idx=None)
    assert ef.shape == (8, 768), ef.shape
    norms = ef.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), norms[:4]
    max_dev = (norms - 1).abs().max().item()
    print(f"    OK  in_planes=768  train logits {tuple(out['logits'].shape)}  "
          f"eval BN unit-norm (max|n-1|={max_dev:.2e})")
    return model


def check_swin_ovli(model, device, img_size, ovli_grid=(8, 4), ovli_dim=256):
    print(f"\n[3] OVLI hook on swin (grid={ovli_grid}, dim={ovli_dim})")
    ovli = OVLIHead(model, in_ch=model.in_planes, proj_dim=ovli_dim,
                    grid=ovli_grid, alpha=0.5, tau=0.05).to(device)
    imgs, labels, views = _make_batch(16, img_size, device, n_pids=4)

    model.train()
    ovli.train()
    # forward populates the hook buffer with the (B,768,H,W) layer4 map
    out = model(imgs, view_idx=None, return_cvfc=False)
    fmap = ovli._buf.get('map', None)
    assert fmap is not None, "OVLI hook did not capture the layer4 map"
    assert fmap.dim() == 4 and fmap.shape[1] == 768, \
        f"hook map must be (B,768,H,W), got {tuple(fmap.shape)}"
    print(f"    hook captured layer4 map: {tuple(fmap.shape)} (NCHW)")

    tok = ovli.tokens_from_cached_map()              # (B, K, ovli_dim)
    K = ovli_grid[0] * ovli_grid[1]
    assert tok.shape == (16, K, ovli_dim), \
        f"tokens should be (16,{K},{ovli_dim}), got {tuple(tok.shape)}"
    tnorm = tok.norm(dim=2)
    assert torch.allclose(tnorm, torch.ones_like(tnorm), atol=1e-4), \
        f"tokens must be per-token L2-normed, got norms {tnorm[0, :3]}"
    print(f"    tokens {tuple(tok.shape)}  per-token unit-norm OK  (K={K})")

    g_ovli = F.normalize(out['bn_feat'].float(), dim=1)
    loss, ps, ns = ovli.loss(g_ovli, tok, labels, views)
    assert torch.isfinite(loss), f"OVLI loss not finite: {loss}"
    assert loss.item() >= 0, loss.item()
    print(f"    OVLI loss={loss.item():.4f} pos={float(ps):.3f} neg={float(ns):.3f}")

    # backprop: gradient must reach a Swin backbone parameter (hook NOT detached).
    model.zero_grad(set_to_none=True)
    ovli.zero_grad(set_to_none=True)
    loss.backward()
    # pick a parameter from the last Swin stage (closest to the hooked map)
    swin_grad_ok = False
    for n, p in model.backbone_swin.swin.named_parameters():
        if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0:
            swin_grad_ok = True
            sample = n
            break
    assert swin_grad_ok, "no gradient reached any Swin parameter (hook detached?)"
    # the OVLI projection must also receive gradient (it is the new learnable set)
    proj_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in ovli.proj.parameters())
    assert proj_grad, "OVLI proj got no gradient"
    print(f"    backward OK  grad reached Swin param '{sample}' and OVLI proj")
    ovli.remove_hook()


def check_swin_ovp(model, device, img_size):
    print("\n[4] OVP loss on swin (feat_dim = in_planes = 768)")
    ovp = OVPMemory(num_pid=4, feat_dim=model.in_planes,
                    momentum=0.2, tau=0.05).to(device)
    imgs, labels, views = _make_batch(16, img_size, device, n_pids=4)
    model.train()
    out = model(imgs, view_idx=None, return_cvfc=False)
    z = F.normalize(out['bn_feat'].float(), dim=1)
    assert z.shape == (16, 768), z.shape
    # first call: bank empty -> loss 0 (cold start), then update + second call > 0
    ovp.update(z.detach(), labels, views)
    loss = ovp.loss(z, labels, views)
    assert torch.isfinite(loss), loss
    print(f"    OVP loss (post one update) = {float(loss):.4f}  feat_dim=768 OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--swin_pretrain', default='',
                    help='path to SOLIDER swin_small.pth (empty -> scratch init)')
    ap.add_argument('--img_size', type=int, nargs=2, default=[256, 128])
    ap.add_argument('--skip_resnet', action='store_true')
    args = ap.parse_args()

    assert torch.cuda.is_available(), \
        "smoke needs CUDA (the SOLIDER Swin forward hard-codes .cuda() for the " \
        "semantic weight)."
    device = 'cuda'
    img_size = tuple(args.img_size)
    torch.manual_seed(0)

    print("=" * 70)
    print(f"Swin-Small backbone smoke  (device={device}, img_size={img_size})")
    print(f"  swin_pretrain = {args.swin_pretrain or '(none -> scratch)'}")
    print("=" * 70)

    if not args.skip_resnet:
        check_resnet(device)

    model = check_swin(device, args.swin_pretrain, img_size)
    # OVLI default token grid is 8x4 (=32 tokens); for a 256x128 input the Swin
    # last-stage map is already 8x4 so adaptive-pool is a no-op (no upsampling).
    check_swin_ovli(model, device, img_size, ovli_grid=(8, 4))
    check_swin_ovp(model, device, img_size)

    print("\n" + "=" * 70)
    print("ALL SWIN BACKBONE SMOKE CHECKS PASSED")
    print("=" * 70)


if __name__ == '__main__':
    main()
