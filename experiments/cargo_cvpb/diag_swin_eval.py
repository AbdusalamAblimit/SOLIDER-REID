# encoding: utf-8
"""
Diagnostic: why does the Swin-Small backbone give near-zero cross-view mAP at
eval while training trains fine?

Loads the EXACT model the CARGO trainer builds (build_model, backbone=swin_small,
semantic_weight=0.2, SOLIDER teacher ckpt), feeds a handful of DISTINCT real CARGO
images (and a random control), and inspects -- in BOTH train() and eval() modes --
every tensor on the eval feature path:

    swin(x) -> outs[-1] (norm3'd last map, NCHW)
            -> avg pool over HxW -> global_feat (B,768)
            -> BatchNorm1d bottleneck -> bn_feat (B,768)
            -> F.normalize -> the eval feature actually used for retrieval.

For each tensor it reports finiteness, per-image pairwise cosine (off-diagonal
mean/min/max), and how much the per-image vectors actually differ.  If the eval
feature has off-diagonal cosine ~1.0 (all images near-identical) -> retrieval is
random -> we have localized the collapse to whichever tensor first goes constant.

Run on a CUDA box w/ the SOLIDER ckpt:
    cd <repo>/experiments/cargo_cvpb
    python3 diag_swin_eval.py --swin_pretrain <repo>/pretrained/swin_small.pth \
        --data_root <repo>/data
"""
import os
import sys
import argparse
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_AFD_REID = os.path.join(_HERE, '..', 'afd_reid')
sys.path.insert(0, _AFD_REID)
sys.path.insert(0, _HERE)

from afd_model import build_model  # noqa: E402  (real Swin/resnet model)


def _args(backbone, swin_pretrain, img_size, semantic_weight=0.2):
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
    ns.swin_semantic_weight = semantic_weight
    ns.img_size = img_size
    return ns


def load_real_images(data_root, img_size, n=8):
    """Grab n distinct CARGO jpgs, resize to img_size, ImageNet-normalize."""
    H, W = img_size
    try:
        from PIL import Image
    except Exception:
        return None
    pats = [
        os.path.join(data_root, 'CARGO', '**', '*.jpg'),
        os.path.join(data_root, 'CARGO', '**', '*.png'),
    ]
    files = []
    for p in pats:
        files += glob.glob(p, recursive=True)
    files = sorted(files)
    if len(files) < n:
        return None
    # spread the picks across the listing so we get different ids/views
    idx = np.linspace(0, len(files) - 1, n).astype(int)
    picks = [files[i] for i in idx]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    out = []
    for f in picks:
        im = Image.open(f).convert('RGB').resize((W, H))
        t = torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1) / 255.0
        t = (t - mean) / std
        out.append(t)
    print(f"  loaded {n} real CARGO images, e.g. {os.path.basename(picks[0])} ... "
          f"{os.path.basename(picks[-1])}")
    return torch.stack(out, 0)


def cos_stats(x):
    """Off-diagonal pairwise cosine of rows of x:(B,D)."""
    xn = F.normalize(x.float(), dim=1)
    g = xn @ xn.t()
    B = g.size(0)
    off = g[~torch.eye(B, dtype=torch.bool, device=g.device)]
    return off.mean().item(), off.min().item(), off.max().item()


def describe(name, x):
    finite = torch.isfinite(x).all().item()
    nan = torch.isnan(x).any().item()
    inf = torch.isinf(x).any().item()
    m, mn, mx = cos_stats(x)
    # per-dim std across the batch: how much do images differ per channel
    chan_std = x.float().std(dim=0)
    print(f"    {name:16s} shape={tuple(x.shape)} finite={finite} "
          f"nan={nan} inf={inf} | pair-cos off-diag mean={m:+.4f} "
          f"min={mn:+.4f} max={mx:+.4f} | mean|val|={x.abs().mean():.4f} "
          f"batch-chan-std(mean)={chan_std.mean():.4e}")
    return m  # off-diag mean cosine


@torch.no_grad()
def probe(model, x, tag):
    """Walk the swin eval path and describe each tensor.  Works on the AFDModel
    swin branch directly (no hooks needed)."""
    swin = model.backbone_swin.swin
    print(f"  [{tag}] model.training={model.training}")
    # full swin forward -> (global_avgpool_feat, outs)
    gfeat_swin, outs = swin(x)
    last = outs[-1]                                   # (B,768,h,w) norm3'd
    describe('outs[-1] map', last.flatten(1))         # flatten spatial+chan
    describe('swin.gfeat', gfeat_swin)                # avgpool inside swin.forward

    # AFDModel path: feat_map -> _pool (avg) -> global_feat -> bottleneck -> bn
    feat_map = model.backbone_swin(x)                 # routes outs[-1] through Identity
    global_feat = model._pool(feat_map)               # (B,768) avg pool
    g_off = describe('global_feat', global_feat)
    bn = model.bottleneck(global_feat)
    bn_off = describe('bn_feat', bn)
    final = F.normalize(bn, dim=1)
    f_off = describe('final(norm bn)', final)
    return {'global': g_off, 'bn': bn_off, 'final': f_off}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--swin_pretrain', default='')
    ap.add_argument('--data_root', default='/root/work/SOLIDER-REID/data')
    ap.add_argument('--img_size', type=int, nargs=2, default=[256, 128])
    ap.add_argument('--semantic_weight', type=float, default=0.2)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs CUDA (swin forward hard-codes .cuda())"
    device = 'cuda'
    img_size = tuple(args.img_size)
    torch.manual_seed(0)

    print("=" * 78)
    print(f"SWIN EVAL DIAGNOSTIC  device={device} img_size={img_size} "
          f"sem_w={args.semantic_weight}")
    print(f"  ckpt={args.swin_pretrain or '(scratch)'}")
    print("=" * 78)

    model = build_model(num_classes=100,
                        args=_args('swin_small', args.swin_pretrain, img_size,
                                   args.semantic_weight)).to(device)

    # inputs: distinct real images (preferred) + a random control
    real = load_real_images(args.data_root, img_size, n=8)
    if real is None:
        print("  [warn] could not load real CARGO images -> using random inputs")
        real = torch.randn(8, 3, *img_size)
    real = real.to(device)
    rand = torch.randn(8, 3, *img_size, device=device)

    print("\n### REAL DISTINCT IMAGES ###")
    print("\n-- TRAIN mode (what the loss sees; BN uses batch stats) --")
    model.train()
    tr = probe(model, real, 'train')
    print("\n-- EVAL mode (what retrieval uses; BN uses running stats) --")
    model.eval()
    ev = probe(model, real, 'eval')

    print("\n### RANDOM CONTROL ###")
    model.eval()
    probe(model, rand, 'eval-rand')

    # BN running-stat sanity: have they ever been updated away from init?
    bb = model.bottleneck
    rm = bb.running_mean
    rv = bb.running_var
    print("\n### BatchNorm1d bottleneck running stats ###")
    print(f"  running_mean: mean={rm.mean():.4e} std={rm.std():.4e} "
          f"min={rm.min():.4e} max={rm.max():.4e}")
    print(f"  running_var : mean={rv.mean():.4e} std={rv.std():.4e} "
          f"min={rv.min():.4e} max={rv.max():.4e}  "
          f"(==1.0 everywhere => never updated / fresh init)")
    print(f"  num_batches_tracked = {int(bb.num_batches_tracked)}")
    print(f"  weight: mean={bb.weight.mean():.4e}  bias: mean={bb.bias.mean():.4e}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print(f"  TRAIN final off-diag cos = {tr['final']:+.4f}  "
          f"(want << 1.0: images distinguishable)")
    print(f"  EVAL  final off-diag cos = {ev['final']:+.4f}  "
          f"(if ~+1.0 -> all images collapse -> random retrieval -> mAP~0)")
    print("=" * 78)


if __name__ == '__main__':
    main()
