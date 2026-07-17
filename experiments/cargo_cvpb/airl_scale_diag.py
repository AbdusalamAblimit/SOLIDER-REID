# encoding: utf-8
"""
AIRL kill-switch -- zero-training aerial-scale-bucketed A->G mAP diagnostic.

Hypothesis under test (new_angle_AIRL.md):
    CARGO aerial->ground error is dominated by the AERIAL crop's low pixel budget
    (small bbox = low resolution = identity physically unresolvable), NOT just a
    view-alignment problem.

Method (NO training; reuse the trained baseline checkpoint + the exact CARGO
eval used by afd_train.eval_market):
    1. Load the baseline model (resnet50 OR swin_small) and its checkpoint. We
       only ever use the eval feature (global BNNeck, L2-normalized) -- for the
       swin OVLI checkpoint the OVLI head is NOT on the eval path, so this is the
       plain strong-Swin global feature (a fair, if slightly conservative, probe).
    2. Read the NATIVE pixel size (PIL .size, BEFORE the eval resize) of every
       aerial query crop and bucket the aerial queries by native bbox AREA into
       --nbuckets equal-count quantile buckets (smallest .. largest).
    3. Extract eval features ONCE for all aerial queries + all ground gallery.
    4. For each aerial-scale bucket, take that bucket's query rows and run the
       SAME market-style mAP (eval_market) against the FULL ground gallery.
       The gallery is identical across buckets, so any per-bucket mAP difference
       is attributable to the AERIAL query scale alone.
    5. (reliability) per query, record the top-1 gallery cosine similarity as a
       confidence; report mean confidence per bucket and a global AUROC of
       "confidence predicts rank-1 correctness" (does the model KNOW when an
       aerial query is unresolvable?).

Verdict:
    gap = (mAP of the highest-scale bucket) - (mAP of the lowest-scale bucket).
    gap > 3-5 mAP  -> aerial scale is a primary error source -> AIRL PASS.
    gap < 3 mAP    -> scale is not the main driver           -> AIRL KILL.

Run on lab-3090 (alongside training is fine; eval-only, no_grad):
    cd /root/work/SOLIDER-REID/experiments/cargo_cvpb
    # strong Swin backbone (preferred):
    PYTHONUNBUFFERED=1 python3 airl_scale_diag.py \
        --backbone swin_small \
        --ckpt /root/work/SOLIDER-REID/log/cargo/cvpb_swin_fix256/model_best.pth \
        --swin_pretrain /root/work/SOLIDER-REID/pretrained/swin_small.pth \
        2>&1 | tee /tmp/airl_scale_diag_swin.log
    # resnet50 baseline (fallback / cross-check):
    PYTHONUNBUFFERED=1 python3 airl_scale_diag.py \
        --backbone resnet50 \
        --ckpt /root/work/SOLIDER-REID/log/cargo/afd_baseline/model_best.pth \
        2>&1 | tee /tmp/airl_scale_diag_r50.log
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

# reuse the afd_reid building blocks unchanged (dataset/model/eval).
# IMPORTANT: only put afd_reid on the path (NOT this cargo_cvpb dir) so that
# `import afd_train` resolves to afd_reid/afd_train.py (the real trainer with
# eval_market), and not the sibling cargo_cvpb/afd_train.py (the OVLI trainer,
# which itself imports `from afd_train import ...` and would circular-import).
_HERE = os.path.dirname(os.path.abspath(__file__))
_AFD_REID = os.path.join(_HERE, '..', 'afd_reid')
sys.path.insert(0, _AFD_REID)

from cargo_dataset import (CARGO, CARGOImageDataset, build_transforms,  # noqa: E402
                           filter_by_view)
from afd_model import build_model  # noqa: E402
from afd_train import eval_market   # noqa: E402  -- the exact market-style eval


def _args(backbone, swin_pretrain, img_size, semantic_weight=0.2):
    """Mirror diag_swin_eval._args so build_model gets the identical config the
    CARGO trainer used (baseline path; use_afd=False)."""
    ns = argparse.Namespace()
    ns.last_stride = 1
    ns.pool = 'gem'
    ns.use_afd = False
    ns.afd_router = False          # baseline eval path: no router
    ns.afd_cvfc = False
    ns.afd_stage = 'layer1'
    ns.router_cond_view = False
    ns.low_r, ns.mid_r, ns.high_drop_p = 0.125, 0.30, 0.5
    ns.backbone = backbone
    ns.swin_pretrain = swin_pretrain
    ns.swin_semantic_weight = semantic_weight
    ns.img_size = img_size
    return ns


@torch.no_grad()
def extract_features(model, loader, device):
    """Return (feats[N,D] float cpu, pids[N] int64, camids[N] int64, paths[N])."""
    model.eval()
    feats, pids, camids, paths = [], [], [], []
    for batch in loader:
        imgs = batch['img'].to(device, non_blocking=True)
        f = model(imgs)                       # eval -> L2-normalized BN feature
        feats.append(f.float().cpu())
        pids.append(batch['pid'])
        camids.append(batch['camid'])
        paths.extend(batch['img_path'])
    feats = torch.cat(feats, 0)
    pids = torch.cat(pids, 0).numpy().astype(np.int64)
    camids = torch.cat(camids, 0).numpy().astype(np.int64)
    return feats, pids, camids, paths


def native_area(path):
    """Native (pre-resize) bbox pixel area + (h, w) from the file. Robust to a
    few flaky synthetic frames (returns None on failure)."""
    try:
        with Image.open(path) as im:
            w, h = im.size           # PIL .size = (width, height)
        return h * w, h, w
    except Exception:
        return None


def auroc(scores, labels):
    """AUROC of `scores` predicting binary `labels` (1=correct). Rank-based
    (Mann-Whitney U); returns nan if only one class present."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    pos = labels == 1
    neg = labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg = {}
    start = 0
    for i, c in enumerate(cnt):
        avg[i] = (start + 1 + start + c) / 2.0
        start += c
    ranks = np.array([avg[i] for i in inv])
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def per_query_top1(qf, gf, q_pids, q_camids, g_pids, g_camids):
    """For each query: (top-1 cosine confidence after junk removal, is rank-1
    correct). Mirrors eval_market's same-(pid,camid) junk removal."""
    qfn = F.normalize(qf, dim=1)
    gfn = F.normalize(gf, dim=1)
    sims = (qfn @ gfn.t()).numpy()             # cosine sim, higher = closer
    conf, correct = [], []
    for i in range(sims.shape[0]):
        s = sims[i].copy()
        junk = (g_pids == q_pids[i]) & (g_camids == q_camids[i])
        s[junk] = -1e9
        j = int(np.argmax(s))
        conf.append(float(s[j]))
        correct.append(int(g_pids[j] == q_pids[i]))
    return np.array(conf), np.array(correct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', default='swin_small',
                    choices=['resnet50', 'swin_small'])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--swin_pretrain', default='')
    ap.add_argument('--data_root', default='/root/work/SOLIDER-REID/data')
    ap.add_argument('--img_size', type=int, nargs=2, default=[256, 128])
    ap.add_argument('--num_classes', type=int, default=2500)  # CARGO proto-1 ALL
    ap.add_argument('--test_batch', type=int, default=64)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--nbuckets', type=int, default=4)
    ap.add_argument('--scale_key', default='area', choices=['area', 'height'],
                    help='native bbox metric to bucket aerial queries by')
    args = ap.parse_args()

    assert torch.cuda.is_available(), "need CUDA"
    device = 'cuda'
    img_size = tuple(args.img_size)

    print('=' * 78)
    print('AIRL kill-switch: aerial-scale-bucketed A->G mAP diagnostic')
    print(f'  backbone={args.backbone}  ckpt={args.ckpt}')
    print(f'  scale_key={args.scale_key}  nbuckets={args.nbuckets}'
          f'  img_size={img_size}')
    print('=' * 78)

    # ---- data ----
    dataset = CARGO(root=args.data_root, verbose=True)
    q_aerial = filter_by_view(dataset.query, 'Aerial')
    g_ground = filter_by_view(dataset.gallery, 'Ground')
    print(f'  aerial queries={len(q_aerial)}  ground gallery={len(g_ground)}')

    # ---- model ----
    model = build_model(num_classes=args.num_classes,
                        args=_args(args.backbone, args.swin_pretrain, img_size)
                        ).to(device)
    sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    res = model.load_state_dict(sd, strict=False)
    # OVLI head keys (ovli.*) are expected-unexpected for the swin OVLI ckpt and
    # are NOT on the eval path; report counts but they are harmless.
    miss = [k for k in res.missing_keys]
    unexp = [k for k in res.unexpected_keys]
    print(f'  load_state_dict: missing={len(miss)} unexpected={len(unexp)}')
    backbone_miss = [k for k in miss if not k.startswith('classifier')]
    if backbone_miss:
        print(f'    [WARN backbone/bottleneck missing keys] {backbone_miss[:8]}')
    if unexp:
        print(f'    [unexpected sample] {unexp[:6]}')
    model.eval()

    tf = build_transforms(is_train=False, img_size=img_size)

    def loader(samples):
        return DataLoader(CARGOImageDataset(samples, tf),
                          batch_size=args.test_batch, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    # ---- extract once ----
    print('  extracting aerial-query features ...')
    qf, qp, qc, qpaths = extract_features(model, loader(q_aerial), device)
    print('  extracting ground-gallery features ...')
    gf, gp, gc, _ = extract_features(model, loader(g_ground), device)

    # sanity: full A->G mAP must match the trained eval (~ the log number)
    full_map, full_cmc, full_minp = eval_market(qf, qp, qc, gf, gp, gc)
    print(f'  [sanity] FULL A->G  mAP={full_map*100:.2f}  R1={full_cmc[0]*100:.2f}'
          f'  mINP={full_minp*100:.2f}  (Nq={len(qp)} Ng={len(gp)})')

    # ---- native scale per aerial query (aligned to qpaths order) ----
    areas, heights, ok_mask = [], [], []
    for p in qpaths:
        na = native_area(p)
        if na is None:
            areas.append(np.nan); heights.append(np.nan); ok_mask.append(False)
        else:
            a, h, w = na
            areas.append(a); heights.append(h); ok_mask.append(True)
    areas = np.array(areas, float)
    heights = np.array(heights, float)
    ok_mask = np.array(ok_mask, bool)
    key = areas if args.scale_key == 'area' else heights
    print(f'  native aerial {args.scale_key}: '
          f'min={np.nanmin(key):.0f} med={np.nanmedian(key):.0f} '
          f'max={np.nanmax(key):.0f}  (readable {ok_mask.sum()}/{len(qpaths)})')

    # ---- quantile buckets (equal count) over readable queries ----
    valid_idx = np.where(ok_mask)[0]
    kv = key[valid_idx]
    qs = np.quantile(kv, np.linspace(0, 1, args.nbuckets + 1))
    # assign each valid query to a bucket [0, nbuckets)
    bucket_of = np.digitize(kv, qs[1:-1], right=False)  # 0..nbuckets-1

    # global reliability (all aerial queries, full gallery)
    conf, correct = per_query_top1(qf, gf, qp, qc, gp, gc)
    glob_auroc = auroc(conf[valid_idx], correct[valid_idx])

    print('\n' + '-' * 78)
    print(f'PER-BUCKET A->G mAP (aerial query scale = native {args.scale_key}; '
          f'gallery = full ground, identical across buckets)')
    print('-' * 78)
    header = (f'{"bucket":>6} | {"n":>4} | {args.scale_key+" range":>22} | '
              f'{"mAP":>6} | {"R1":>6} | {"R5":>6} | {"mINP":>6} | '
              f'{"meanConf":>8} | {"R1acc":>6}')
    print(header)
    rows = []
    for b in range(args.nbuckets):
        sel = valid_idx[bucket_of == b]
        if len(sel) == 0:
            continue
        bqf = qf[sel]; bqp = qp[sel]; bqc = qc[sel]
        mAP, cmc, minp = eval_market(bqf, bqp, bqc, gf, gp, gc)
        lo, hi = key[sel].min(), key[sel].max()
        mc = float(conf[sel].mean())
        r1acc = float(correct[sel].mean()) * 100
        rng = f'{lo:.0f}-{hi:.0f}'
        rows.append((b, len(sel), mAP * 100, cmc[0] * 100,
                     cmc[4] * 100 if len(cmc) > 4 else float('nan'),
                     minp * 100, mc, r1acc))
        print(f'{b:>6} | {len(sel):>4} | {rng:>22} | {mAP*100:>6.2f} | '
              f'{cmc[0]*100:>6.2f} | '
              f'{(cmc[4]*100 if len(cmc)>4 else float("nan")):>6.2f} | '
              f'{minp*100:>6.2f} | {mc:>8.4f} | {r1acc:>6.1f}')

    # ---- verdict ----
    print('-' * 78)
    if len(rows) >= 2:
        lo_map = rows[0][2]              # lowest-scale bucket mAP
        hi_map = rows[-1][2]             # highest-scale bucket mAP
        gap = hi_map - lo_map
        # also report the max spread across any two buckets (robustness)
        maps = [r[2] for r in rows]
        spread = max(maps) - min(maps)
        print(f'lowest-scale bucket  mAP = {lo_map:.2f}  (b{rows[0][0]})')
        print(f'highest-scale bucket mAP = {hi_map:.2f}  (b{rows[-1][0]})')
        print(f'GAP (high - low)         = {gap:+.2f} mAP')
        print(f'max spread (any buckets) = {spread:.2f} mAP')
        print(f'reliability AUROC (conf -> R1 correct, all aerial q) = '
              f'{glob_auroc:.3f}')
        print('-' * 78)
        if gap >= 3.0:
            print(f'VERDICT: PASS  (gap {gap:+.2f} >= +3.0)  -> aerial low pixel '
                  f'budget IS a primary A->G error source; AIRL angle is worth '
                  f'pursuing.')
        else:
            print(f'VERDICT: KILL  (gap {gap:+.2f} < +3.0)  -> aerial scale is NOT '
                  f'the dominant A->G error driver; do NOT pursue AIRL.')
    else:
        print('VERDICT: INCONCLUSIVE (need >=2 non-empty buckets)')
    print('=' * 78)


if __name__ == '__main__':
    main()
