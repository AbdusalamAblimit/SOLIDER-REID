# encoding: utf-8
"""
Band-ablation analysis on a TRAINED baseline -- the confound diagnostic.

Loads a trained AFD-ReID baseline (model_best.pth), then for A<->G cross-view
retrieval, applies a frequency-band filter to BOTH query and gallery input images
and measures mAP / Rank-1 / mINP for each band condition:

    orig      : unmodified image
    no_high   : drop high band  (keep low+mid)        <- "remove high-freq texture"
    no_low    : drop low band   (keep mid+high)
    only_low  : keep low band only
    only_high : keep high band only

Reported separately for Aerial-as-query (A->G) and Ground-as-query (G->A).

Confound reading (★ what AFD-ReID claims):
    If removing the high band hurts GROUND-as-query much more than AERIAL-as-query
    (because aerial high-freq is already unreliable / compressed), the altitude-
    frequency entanglement is real -> AFD modules are justified.
    If "no_high" hurts both views equally, or aerial actually *relies* on high-freq
    too, the confound is weak -> reconsider.

Run on lab-3090:
    cd /root/work/SOLIDER-REID/experiments/afd_reid
    python band_analysis.py \
        --data_root /root/work/SOLIDER-REID/data \
        --ckpt /root/work/SOLIDER-REID/log/cargo/afd_baseline/model_best.pth
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cargo_dataset import CARGO, build_transforms, filter_by_view
from afd_model import build_model
from afd_train import eval_market


# --------------------------------------------------------------------------- #
# band filtering on a normalized image tensor
# --------------------------------------------------------------------------- #
def fft_band_filter(x, mode, low_r=0.125, mid_r=0.30):
    """x:(3,H,W) tensor (already normalized). Return band-filtered tensor.

    Uses centered rectangular FFT masks matching afd_model.decompose_bands:
      low  = central box (<= low_r)
      mid  = ring (low_r..mid_r)
      high = outside mid_r
    """
    if mode == 'orig':
        return x
    H, W = x.shape[-2:]
    f = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
    cy, cx = H // 2, W // 2
    ry1, rx1 = max(1, int(H * low_r)), max(1, int(W * low_r))
    ry2, rx2 = max(ry1 + 1, int(H * mid_r)), max(rx1 + 1, int(W * mid_r))

    low = torch.zeros(H, W)
    low[cy - ry1:cy + ry1, cx - rx1:cx + rx1] = 1.0
    midbox = torch.zeros(H, W)
    midbox[cy - ry2:cy + ry2, cx - rx2:cx + rx2] = 1.0
    mid = midbox - low
    high = 1.0 - midbox

    sel = {
        'no_high':   low + mid,
        'no_low':    mid + high,
        'only_low':  low,
        'only_high': high,
    }[mode]
    f = f * sel.view(1, H, W)
    return torch.fft.ifft2(torch.fft.ifftshift(f, dim=(-2, -1)), dim=(-2, -1)).real


class BandDataset(Dataset):
    """Eval dataset that applies a frequency band filter AFTER normalization."""

    def __init__(self, samples, mode, img_size=(256, 128), low_r=0.125, mid_r=0.30):
        self.samples = samples
        self.mode = mode
        self.low_r = low_r
        self.mid_r = mid_r
        self.tf = build_transforms(is_train=False, img_size=img_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['img_path']).convert('RGB')
        x = self.tf(img)
        x = fft_band_filter(x, self.mode, self.low_r, self.mid_r)
        return {'img': x, 'pid': s['pid'], 'camid': s['camid'], 'view': s['view']}


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    feats, pids, camids = [], [], []
    for b in loader:
        f = model(b['img'].to(device), view_idx=None)
        feats.append(f.cpu())
        pids.append(b['pid'])
        camids.append(b['camid'])
    return (torch.cat(feats), torch.cat(pids).numpy(), torch.cat(camids).numpy())


def eval_dir(model, q_samples, g_samples, mode, args, device):
    ql = DataLoader(BandDataset(q_samples, mode, tuple(args.img_size)),
                    batch_size=args.test_batch, num_workers=args.workers,
                    pin_memory=True)
    gl = DataLoader(BandDataset(g_samples, mode, tuple(args.img_size)),
                    batch_size=args.test_batch, num_workers=args.workers,
                    pin_memory=True)
    qf, qp, qc = extract(model, ql, device)
    gf, gp, gc = extract(model, gl, device)
    mAP, cmc, mINP = eval_market(qf, qp, qc, gf, gp, gc)
    return mAP * 100, cmc[0] * 100, mINP * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='/root/work/SOLIDER-REID/data')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--img_size', type=int, nargs=2, default=[256, 128])
    ap.add_argument('--test_batch', type=int, default=128)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--low_r', type=float, default=0.125)
    ap.add_argument('--mid_r', type=float, default=0.30)
    ap.add_argument('--pool', default='gem', choices=['gem', 'avg'])
    ap.add_argument('--last_stride', type=int, default=1)
    args = ap.parse_args()
    args.use_afd = False   # analysis runs on the baseline backbone
    device = 'cuda'

    dataset = CARGO(root=args.data_root, verbose=True)
    model = build_model(dataset.num_train_pids, args).to(device)
    state = torch.load(args.ckpt, map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)} "
              f"(ok if only AFD keys differ)")
    print(f"=> loaded checkpoint: {args.ckpt}")

    q_aerial = filter_by_view(dataset.query, 'Aerial')
    q_ground = filter_by_view(dataset.query, 'Ground')
    g_aerial = filter_by_view(dataset.gallery, 'Aerial')
    g_ground = filter_by_view(dataset.gallery, 'Ground')

    modes = ['orig', 'no_high', 'no_low', 'only_low', 'only_high']
    print("\n" + "=" * 72)
    print("BAND ABLATION  (mAP / R1 / mINP), low_r=%.3f mid_r=%.3f"
          % (args.low_r, args.mid_r))
    print("=" * 72)
    header = f"{'band':10s} | {'A->G mAP':>9s} {'R1':>6s} {'mINP':>6s} | " \
             f"{'G->A mAP':>9s} {'R1':>6s} {'mINP':>6s}"
    print(header)
    print("-" * len(header))

    table = {}
    for mode in modes:
        a_map, a_r1, a_inp = eval_dir(model, q_aerial, g_ground, mode, args, device)
        g_map, g_r1, g_inp = eval_dir(model, q_ground, g_aerial, mode, args, device)
        table[mode] = (a_map, g_map)
        print(f"{mode:10s} | {a_map:9.2f} {a_r1:6.2f} {a_inp:6.2f} | "
              f"{g_map:9.2f} {g_r1:6.2f} {g_inp:6.2f}")

    print("=" * 72)
    # confound summary: how much does dropping high-band hurt each direction?
    a_drop = table['orig'][0] - table['no_high'][0]   # A->G mAP loss from no_high
    g_drop = table['orig'][1] - table['no_high'][1]   # G->A mAP loss from no_high
    print("CONFOUND SUMMARY (effect of removing HIGH band, mAP drop):")
    print(f"  A->G (aerial-as-query):  {a_drop:+.2f}")
    print(f"  G->A (ground-as-query):  {g_drop:+.2f}")
    if g_drop > a_drop + 1.0:
        print("  => Ground relies on high-freq MORE than Aerial. "
              "Altitude-frequency entanglement SUPPORTED.")
    elif a_drop > g_drop + 1.0:
        print("  => Aerial relies on high-freq more (unexpected). "
              "Reconsider the confound direction.")
    else:
        print("  => Similar reliance on high-freq across views. "
              "Confound WEAK at this band split.")
    print("=" * 72)


if __name__ == '__main__':
    main()
