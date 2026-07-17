# encoding: utf-8
"""
Load the TRAINED (collapsed) cvpb_swin_ovli checkpoint and confirm whether its
eval features are degenerate (all images -> ~same vector), which is the direct
cause of mAP~0.03.  Also reports BN running stats & a few backbone weight norms
so we can tell collapse (weights blew up / went constant) from a forward bug.

Run on lab-3090:
    cd <repo>/experiments/cargo_cvpb
    python3 diag_swin_ckpt.py \
        --ckpt /root/work/SOLIDER-REID/log/cargo/cvpb_swin_ovli/model_best.pth \
        --swin_pretrain /root/work/SOLIDER-REID/pretrained/swin_small.pth \
        --data_root /root/work/SOLIDER-REID/data
"""
import os
import sys
import argparse
import glob

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_AFD_REID = os.path.join(_HERE, '..', 'afd_reid')
sys.path.insert(0, _AFD_REID)
sys.path.insert(0, _HERE)
from afd_model import build_model  # noqa: E402
from diag_swin_eval import load_real_images, cos_stats, describe, _args  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--swin_pretrain', default='')
    ap.add_argument('--data_root', default='/root/work/SOLIDER-REID/data')
    ap.add_argument('--img_size', type=int, nargs=2, default=[256, 128])
    ap.add_argument('--num_classes', type=int, default=2500)  # CARGO protocol-1 ALL
    args = ap.parse_args()

    assert torch.cuda.is_available()
    device = 'cuda'
    img_size = tuple(args.img_size)

    print("=" * 78)
    print(f"LOAD TRAINED CKPT: {args.ckpt}")
    print("=" * 78)

    model = build_model(num_classes=args.num_classes,
                        args=_args('swin_small', args.swin_pretrain, img_size, 0.2)).to(device)
    sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    res = model.load_state_dict(sd, strict=False)
    print(f"  load_state_dict missing={len(res.missing_keys)} "
          f"unexpected={len(res.unexpected_keys)}")
    if res.missing_keys[:5]:
        print(f"    missing e.g.: {res.missing_keys[:5]}")
    if res.unexpected_keys[:5]:
        print(f"    unexpected e.g.: {res.unexpected_keys[:5]}")

    real = load_real_images(args.data_root, img_size, n=8)
    if real is None:
        real = torch.randn(8, 3, *img_size)
    real = real.to(device)

    model.eval()
    print("\n-- EVAL features from the TRAINED checkpoint --")
    with torch.no_grad():
        swin = model.backbone_swin.swin
        _g, outs = swin(real)
        describe('outs[-1] map', outs[-1].flatten(1))
        feat_map = model.backbone_swin(real)
        gfeat = model._pool(feat_map)
        describe('global_feat', gfeat)
        bn = model.bottleneck(gfeat)
        describe('bn_feat', bn)
        final = F.normalize(bn, dim=1)
        foff, fmin, fmax = cos_stats(final)
        describe('final(norm bn)', final)

    bb = model.bottleneck
    print("\n### BN bottleneck running stats (trained) ###")
    print(f"  running_mean: mean={bb.running_mean.mean():.4e} "
          f"absmax={bb.running_mean.abs().max():.4e}")
    print(f"  running_var : mean={bb.running_var.mean():.4e} "
          f"min={bb.running_var.min():.4e} max={bb.running_var.max():.4e}")
    print(f"  num_batches_tracked = {int(bb.num_batches_tracked)}")
    print(f"  weight absmax={bb.weight.abs().max():.4e}  "
          f"finite={torch.isfinite(bb.weight).all().item()}")

    # backbone weight health: any NaN/Inf or exploded norms?
    print("\n### Swin backbone weight health ###")
    bad = 0
    big = []
    for n, p in model.backbone_swin.swin.named_parameters():
        if not torch.isfinite(p).all():
            bad += 1
            if len(big) < 8:
                big.append(f"NONFINITE {n}")
        elif p.abs().max() > 50:
            if len(big) < 8:
                big.append(f"{n}: absmax={p.abs().max():.1f}")
    print(f"  non-finite param tensors: {bad}")
    for s in big:
        print(f"    {s}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print(f"  EVAL final off-diag cos: mean={foff:+.4f} min={fmin:+.4f} max={fmax:+.4f}")
    print("  ~+1.0 => all images collapse to one vector => retrieval random => mAP~0")
    print("=" * 78)


if __name__ == '__main__':
    main()
