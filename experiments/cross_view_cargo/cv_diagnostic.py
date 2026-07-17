# encoding: utf-8
"""
CVCL kill-switch (NO TRAINING): test the *cross-view positive scarcity* confound on CARGO.

Loads the baseline resnet50-BoT checkpoint (model_best.pth, mean A<->G mAP 32.48%)
and reports two evidence blocks, then a PASS/FAIL verdict.

(a) BATCH SAMPLING STATISTICS
    - per-pid aerial(cam1-5)/ground(cam6-13) image-count distribution over CARGO train
    - fraction of train pids that are "dual-view" (have >=1 aerial AND >=1 ground image)
    - simulate the ACTUAL RandomIdentitySampler(P=16, K=4) used in afd_train.py for
      1000 batches and measure, per anchor, whether the batch contains at least one
      OPPOSITE-view same-id positive. Report the mean fraction of anchors with an
      opposite-view positive available in-batch.

(b) BASELINE FEATURE DISTANCES (l2-normalized BN feature, cosine + euclidean)
    On a balanced sample of train images, measure:
      - same-id  same-view   distance (easy positive)
      - same-id  cross-view  distance (the hard, view-bridging positive)
      - diff-id  cross-view  nearest  (hard negative across views)
    The confound predicts: same-id cross-view >> same-id same-view, and
    same-id cross-view ~ diff-id cross-view hard-neg (positives drowned by negatives).

(c) VERDICT
    PASS  = in-batch opposite-view positive availability is LOW (< 70%)
            AND same-id cross-view distance clearly > same-id same-view distance
            AND same-id cross-view distance is close to the diff-id cross-view hard-neg.
    FAIL  = positives plentiful in batch OR cross-view ~ same-view (model already
            view-invariant) -> confound dead, pivot to another angle.

Run on lab-3090 (single GPU, ~1-2 min):
    cd /root/work/SOLIDER-REID/experiments/cross_view_cargo
    PYTHONUNBUFFERED=1 python cv_diagnostic.py \
        --data_root /root/work/SOLIDER-REID/data \
        --ckpt /root/work/SOLIDER-REID/log/cargo/afd_baseline/model_best.pth
"""
import os
import sys
import argparse
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# reuse the existing afd_reid interfaces (do NOT reimplement dataset / model)
_HERE = os.path.dirname(os.path.abspath(__file__))
_AFD = os.path.normpath(os.path.join(_HERE, '..', 'afd_reid'))
sys.path.insert(0, _AFD)
from cargo_dataset import (CARGO, CARGOImageDataset, build_transforms,  # noqa: E402
                          RandomIdentitySampler)
from afd_model import build_model  # noqa: E402

VIEW2IDX = {'Aerial': 0, 'Ground': 1}


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #
def load_baseline(ckpt_path, num_classes, device):
    """Build AFDModel(use_afd=False) == plain resnet50 BoT and load the state_dict."""
    class _A:  # argparse-like namespace; baseline => use_afd off
        last_stride = 1
        pool = 'gem'
        use_afd = False
        afd_router = False
        afd_cvfc = False
        afd_stage = 'layer1'
        router_cond_view = False
    model = build_model(num_classes, _A()).to(device)
    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state and \
            not any(k.startswith(('classifier', 'bottleneck', 'layer')) for k in state):
        state = state['state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    # classifier is trained for the TRAIN id space; fine for feature extraction either way.
    if missing:
        print(f"[load] missing keys ({len(missing)}): {missing[:6]}{' ...' if len(missing) > 6 else ''}")
    if unexpected:
        print(f"[load] unexpected keys ({len(unexpected)}): {unexpected[:6]}{' ...' if len(unexpected) > 6 else ''}")
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# (a) batch sampling statistics
# --------------------------------------------------------------------------- #
def batch_sampling_stats(train_samples, P=16, K=4, n_batches=1000, seed=0):
    """Per-pid view counts + dual-view ratio + simulated in-batch opposite-view
    positive availability for the REAL RandomIdentitySampler."""
    # per-pid aerial/ground counts and per-pid global index lists by view
    a_count = defaultdict(int)
    g_count = defaultdict(int)
    view_of = np.empty(len(train_samples), dtype=np.int8)
    pid_of = np.empty(len(train_samples), dtype=np.int64)
    for idx, s in enumerate(train_samples):
        v = VIEW2IDX[s['view']]
        view_of[idx] = v
        pid_of[idx] = s['pid']
        if v == 0:
            a_count[s['pid']] += 1
        else:
            g_count[s['pid']] += 1

    all_pids = sorted({s['pid'] for s in train_samples})
    n_pid = len(all_pids)
    dual = [p for p in all_pids if a_count[p] > 0 and g_count[p] > 0]
    aerial_only = [p for p in all_pids if a_count[p] > 0 and g_count[p] == 0]
    ground_only = [p for p in all_pids if g_count[p] > 0 and a_count[p] == 0]

    a_per = np.array([a_count[p] for p in all_pids])
    g_per = np.array([g_count[p] for p in all_pids])

    print("=" * 72)
    print("(a) BATCH SAMPLING STATISTICS")
    print("-" * 72)
    print(f"  train images={len(train_samples)}  pids={n_pid}")
    print(f"  aerial imgs/pid: mean={a_per.mean():.2f} med={np.median(a_per):.0f} "
          f"min={a_per.min()} max={a_per.max()}")
    print(f"  ground imgs/pid: mean={g_per.mean():.2f} med={np.median(g_per):.0f} "
          f"min={g_per.min()} max={g_per.max()}")
    print(f"  DUAL-view pids (have aerial AND ground): {len(dual)}/{n_pid} "
          f"= {100.0 * len(dual) / n_pid:.1f}%")
    print(f"  aerial-only pids: {len(aerial_only)}/{n_pid} = {100.0 * len(aerial_only) / n_pid:.1f}%")
    print(f"  ground-only pids: {len(ground_only)}/{n_pid} = {100.0 * len(ground_only) / n_pid:.1f}%")

    # --- simulate the REAL sampler ---
    # The sampler builds K-instance chunks per pid then packs P pids per batch.
    # We iterate it for ~n_batches batches and, for every anchor in a batch,
    # check whether some other sample in the SAME batch is same-id & opposite-view.
    random.seed(seed)
    np.random.seed(seed)
    sampler = RandomIdentitySampler(train_samples, batch_size=P * K, num_instances=K)
    bs = P * K

    anchors_total = 0
    anchors_with_opp_pos = 0          # >=1 opposite-view same-id positive in batch
    anchors_with_any_pos = 0          # >=1 same-id positive at all (sanity)
    dual_anchors_total = 0            # anchors whose pid is dual-view (opp-view exists in dataset)
    dual_anchors_with_opp_pos = 0     # of those, how many actually got one in-batch
    batches_done = 0
    # opposite-view positives per anchor (distribution)
    opp_pos_counts = []

    # loop sampler epochs until we have enough batches
    while batches_done < n_batches:
        flat = list(iter(sampler))
        for b0 in range(0, len(flat) - bs + 1, bs):
            if batches_done >= n_batches:
                break
            idxs = flat[b0:b0 + bs]
            bp = pid_of[idxs]
            bv = view_of[idxs]
            for j in range(bs):
                pid_j, view_j = bp[j], bv[j]
                same_pid = (bp == pid_j)
                same_pid[j] = False                  # exclude the anchor itself
                opp_view = (bv != view_j)
                opp_pos = same_pid & opp_view
                n_opp = int(opp_pos.sum())
                anchors_total += 1
                anchors_with_any_pos += int(same_pid.any())
                anchors_with_opp_pos += int(n_opp > 0)
                opp_pos_counts.append(n_opp)
                if a_count[pid_j] > 0 and g_count[pid_j] > 0:  # dual-view pid
                    dual_anchors_total += 1
                    dual_anchors_with_opp_pos += int(n_opp > 0)
            batches_done += 1

    opp_pos_counts = np.array(opp_pos_counts)
    frac_opp = 100.0 * anchors_with_opp_pos / max(1, anchors_total)
    frac_any = 100.0 * anchors_with_any_pos / max(1, anchors_total)
    frac_dual = (100.0 * dual_anchors_with_opp_pos / max(1, dual_anchors_total)
                 if dual_anchors_total else float('nan'))
    print("-" * 72)
    print(f"  simulated {batches_done} batches of the REAL RandomIdentitySampler "
          f"(P={P}, K={K}, bs={bs})")
    print(f"  anchors with >=1 OPPOSITE-view same-id positive in-batch : {frac_opp:.1f}%")
    print(f"    (sanity) anchors with >=1 same-id positive (any view)  : {frac_any:.1f}%")
    print(f"  among DUAL-view-pid anchors only, got opp-view positive   : {frac_dual:.1f}%")
    print(f"  opposite-view positives per anchor: mean={opp_pos_counts.mean():.2f} "
          f"(0 for {100.0 * (opp_pos_counts == 0).mean():.1f}% of anchors)")
    return {'frac_opp_pos': frac_opp, 'dual_ratio': 100.0 * len(dual) / n_pid,
            'frac_opp_pos_dual': frac_dual}


# --------------------------------------------------------------------------- #
# (b) baseline feature distances
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_feats(model, samples, args, device):
    """Return (feat[N,D] l2-normalized, pids[N], views[N]) in the SAME order as samples."""
    tf = build_transforms(is_train=False, img_size=tuple(args.img_size))
    ds = CARGOImageDataset(samples, tf)
    loader = DataLoader(ds, batch_size=args.test_batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    feats, pids, views = [], [], []
    for batch in loader:
        imgs = batch['img'].to(device, non_blocking=True)
        f = model(imgs)                       # eval-mode => already F.normalize'd BN feat
        feats.append(f.cpu())
        pids.append(batch['pid'])
        views.extend(VIEW2IDX[v] for v in batch['view'])
    feats = torch.cat(feats, 0)
    feats = F.normalize(feats, dim=1)          # ensure unit norm (idempotent)
    pids = torch.cat(pids, 0).numpy()
    views = np.array(views, dtype=np.int8)
    return feats, pids, views


def sample_balanced_dualview(train_samples, max_pids=150, per_view=4, seed=0):
    """Pick dual-view pids and, for each, a few aerial + a few ground images,
    so cross-view positive pairs actually exist in the probe set."""
    by_pid_view = defaultdict(lambda: {0: [], 1: []})
    for s in train_samples:
        by_pid_view[s['pid']][VIEW2IDX[s['view']]].append(s)
    dual = [p for p in by_pid_view if by_pid_view[p][0] and by_pid_view[p][1]]
    rng = random.Random(seed)
    rng.shuffle(dual)
    dual = dual[:max_pids]
    picked = []
    for p in dual:
        for v in (0, 1):
            pool = by_pid_view[p][v]
            rng.shuffle(pool)
            picked.extend(pool[:per_view])
    return picked, len(dual)


def feature_distance_stats(model, train_samples, args, device, seed=0):
    probe, n_dual = sample_balanced_dualview(
        train_samples, max_pids=args.probe_pids, per_view=args.probe_per_view, seed=seed)
    feats, pids, views = extract_feats(model, probe, args, device)
    N = feats.size(0)
    # cosine distance = 1 - cos sim ; also report euclidean on unit vectors (= sqrt(2*cos_dist))
    sim = (feats @ feats.t()).numpy()
    cos_dist = 1.0 - sim
    pid_eq = pids[:, None] == pids[None, :]
    view_eq = views[:, None] == views[None, :]
    eye = np.eye(N, dtype=bool)

    same_id_same_view = pid_eq & view_eq & ~eye
    same_id_cross_view = pid_eq & ~view_eq
    diff_id_cross_view = ~pid_eq & ~view_eq

    def mean_of(mask):
        vals = cos_dist[mask]
        return float(vals.mean()) if vals.size else float('nan')

    d_ssv = mean_of(same_id_same_view)
    d_scv = mean_of(same_id_cross_view)

    # hard negative: for each query, the NEAREST diff-id cross-view gallery sample.
    big = cos_dist.copy()
    big[~diff_id_cross_view] = np.inf
    nn_neg = big.min(axis=1)
    nn_neg = nn_neg[np.isfinite(nn_neg)]
    d_dcv_hard = float(nn_neg.mean()) if nn_neg.size else float('nan')
    # mean diff-id cross-view (for reference)
    d_dcv_mean = mean_of(diff_id_cross_view)

    # also: for each anchor with a cross-view positive, is its hardest same-id
    # cross-view positive FARTHER than its nearest diff-id cross-view negative?
    pos_cv = cos_dist.copy(); pos_cv[~same_id_cross_view] = -np.inf
    hardest_pos = pos_cv.max(axis=1)                       # largest dist same-id cross-view
    neg_cv = cos_dist.copy(); neg_cv[~diff_id_cross_view] = np.inf
    nearest_neg = neg_cv.min(axis=1)
    valid = np.isfinite(hardest_pos) & (hardest_pos > -1e30) & np.isfinite(nearest_neg)
    violated = (hardest_pos[valid] > nearest_neg[valid]).mean() * 100 if valid.any() else float('nan')

    cos_to_eucl = lambda d: float(np.sqrt(max(0.0, 2.0 * d)))
    print("=" * 72)
    print("(b) BASELINE FEATURE DISTANCES  (l2-normalized BN feature)")
    print("-" * 72)
    print(f"  probe set: {N} imgs from {n_dual} dual-view pids "
          f"(<= {args.probe_pids} pids x {args.probe_per_view}/view)")
    print(f"  [cosine distance = 1 - cos sim; euclidean on unit vecs in brackets]")
    print(f"  same-id  same-view   : cos={d_ssv:.4f}  [eucl={cos_to_eucl(d_ssv):.4f}]   (easy positive)")
    print(f"  same-id  cross-view  : cos={d_scv:.4f}  [eucl={cos_to_eucl(d_scv):.4f}]   (HARD view-bridging positive)")
    print(f"  diff-id  cross-view  : cos={d_dcv_mean:.4f}  [eucl={cos_to_eucl(d_dcv_mean):.4f}]   (mean negative)")
    print(f"  diff-id  cross-view  : cos={d_dcv_hard:.4f}  [eucl={cos_to_eucl(d_dcv_hard):.4f}]   (NEAREST = hard negative)")
    print("-" * 72)
    gap_pos = d_scv - d_ssv
    margin_to_neg = d_dcv_hard - d_scv
    print(f"  cross-view positive penalty  (scv - ssv)        = {gap_pos:+.4f}")
    print(f"  positive-vs-hardneg margin   (dcv_hard - scv)   = {margin_to_neg:+.4f}  "
          f"(<=0 means positives drown in negatives)")
    print(f"  anchors whose hardest cross-view positive is FARTHER than nearest "
          f"cross-view negative: {violated:.1f}%")
    return {'d_ssv': d_ssv, 'd_scv': d_scv, 'd_dcv_hard': d_dcv_hard,
            'gap_pos': gap_pos, 'margin_to_neg': margin_to_neg, 'violated': violated}


# --------------------------------------------------------------------------- #
# (c) verdict
# --------------------------------------------------------------------------- #
def verdict(a_stats, b_stats, opp_thresh=70.0):
    print("=" * 72)
    print("(c) VERDICT")
    print("-" * 72)
    c1 = a_stats['frac_opp_pos'] < opp_thresh
    c2 = b_stats['gap_pos'] > 0 and b_stats['d_scv'] > 1.15 * b_stats['d_ssv']
    # "close to hard-neg": cross-view positive within a small margin of the hard negative
    c3 = b_stats['margin_to_neg'] <= 0.10 * max(b_stats['d_scv'], 1e-6)

    print(f"  C1 in-batch opp-view positives scarce (<{opp_thresh:.0f}%): "
          f"{a_stats['frac_opp_pos']:.1f}%  -> {'YES' if c1 else 'no'}")
    print(f"  C2 same-id cross-view dist clearly > same-view "
          f"(scv {b_stats['d_scv']:.4f} > 1.15*ssv {1.15 * b_stats['d_ssv']:.4f}): "
          f"-> {'YES' if c2 else 'no'}")
    print(f"  C3 cross-view positive close to hard negative "
          f"(margin {b_stats['margin_to_neg']:+.4f} <= 10% of scv): "
          f"-> {'YES' if c3 else 'no'}")
    print("-" * 72)
    if c1 and c2 and c3:
        print("  >>> PASS: cross-view positive scarcity confound is REAL.")
        print("      In-batch cross-view positives are scarce AND the baseline keeps")
        print("      same-id cross-view pairs far apart (near the hard-negative wall).")
        print("      -> proceed to build VC-PK sampler + CV-triplet (cv_train.py --cv).")
    elif c1 and (c2 or c3):
        print("  >>> WEAK-PASS: positives are scarce in-batch and the feature geometry")
        print("      partially supports the confound (C2/C3 mixed). Worth one --cv run,")
        print("      but expect a modest gain; read the numbers before committing.")
    else:
        print("  >>> FAIL: confound not supported (positives plentiful OR cross-view")
        print("      distances ~ same-view). Model already view-invariant enough.")
        print("      -> pivot to another angle (accessory de-confound / T2I-binding).")
    print("=" * 72)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='/root/work/SOLIDER-REID/data')
    ap.add_argument('--ckpt',
                    default='/root/work/SOLIDER-REID/log/cargo/afd_baseline/model_best.pth')
    ap.add_argument('--img_size', type=int, nargs=2, default=[256, 128])
    ap.add_argument('--P', type=int, default=16)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--n_batches', type=int, default=1000)
    ap.add_argument('--test_batch', type=int, default=128)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--probe_pids', type=int, default=150)
    ap.add_argument('--probe_per_view', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("#" * 72)
    print("CVCL kill-switch: cross-view positive scarcity confound on CARGO")
    print(f"  data_root={args.data_root}")
    print(f"  ckpt={args.ckpt}")
    print(f"  device={device}")
    print("#" * 72)

    dataset = CARGO(root=args.data_root, verbose=True)

    # (a) sampling stats (no model needed)
    a_stats = batch_sampling_stats(dataset.train, P=args.P, K=args.K,
                                   n_batches=args.n_batches, seed=args.seed)

    # load baseline for (b)
    if not os.path.isfile(args.ckpt):
        print(f"\n[ERROR] checkpoint not found: {args.ckpt}\n"
              "        (a) printed above; skipping feature-distance block (b).")
        return
    model = load_baseline(args.ckpt, dataset.num_train_pids, device)

    # (b) feature distances
    b_stats = feature_distance_stats(model, dataset.train, args, device, seed=args.seed)

    # (c) verdict
    verdict(a_stats, b_stats)


if __name__ == '__main__':
    main()
