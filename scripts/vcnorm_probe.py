#!/usr/bin/env python
"""VC-Norm prerequisite probe (no training).

QUESTION
--------
VC-Norm (occlusion-as-domain-factor, visibility-conditioned normalization)
treats occlusion as a DOMAIN FACTOR and aligns per-part-token normalization
statistics so visible-part representations are robust to occluded/un-occluded
conditions. The whole idea only has fuel if occlusion actually produces a
*separable distribution shift* in the per-part-token normalization statistics.

This script measures that shift on Occluded-ReID using the Market-trained
exp260b model (Swin-Base + PSG + LGPA + GCN512), WITHOUT any training.

WHAT IS A "PER-PART TOKEN" HERE
-------------------------------
exp260b's part branch (SkeletonGCNHead) samples one feature token per COCO
keypoint from the PSG-modulated Stage-3 feature map (bilinear sample), then
runs a Skeleton GCN. We capture both:
  - pre-GCN  : raw sampled per-keypoint tokens (what GCN/Norm consumes)
  - post-GCN : GCN-enhanced per-keypoint tokens (aux_data['kp_feats'])
For each keypoint k, the per-image token x_k is a (C,) vector. We treat
keypoint-k's token across many images as a population.

VISIBILITY SPLIT
----------------
Pose confidence score (pose_dict['scores'][:,0,k]) == GT-ish visibility for
keypoint k (ViTPose conf). For each keypoint we split its token population by
score into HIGH-VIS (score >= hi_thr) vs LOW-VIS / occluded (score <= lo_thr).

DISTRIBUTION DISTANCE
---------------------
For each keypoint k we compute, between HIGH-VIS and LOW-VIS token populations:
  - Gaussian KL (symmetric) on per-channel diagonal-Gaussian fits:
        KL_sym = 0.5*(KL(P||Q) + KL(Q||P)), summed over channels (diagonal).
    This is exactly the "normalization statistic" (mean/var per channel) shift
    that VC-Norm would try to align.
  - 2-Wasserstein on per-channel diagonal Gaussians (mean+std shift), as a
    scale-interpretable companion to KL.
  - Mean-shift L2 (||mu_hi - mu_lo||) and a normalized version
    (||mu_hi - mu_lo|| / pooled_std) = effect-size of the first moment.
  - Linear separability AUC: logistic-regression-free, use Fisher LDA direction
    + ROC AUC of the 1-D projection (5-fold) as a separability proxy.

VERDICT
-------
KL ~ 0 and AUC ~ 0.5 across parts -> occlusion does NOT cause separable
normalization-statistic shift -> no fuel -> KILL.
KL clearly large and AUC >> 0.5 -> occlusion is an alignable domain factor ->
VC-Norm has headroom (next step: 1-2d dual-forward Market 30ep, target
Occ-ReID mAP > 88.0 baseline).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from datasets.occluded_reid import OccludedREID
from datasets.pose_dataset import PoseImageDataset, pose_val_collate_fn
from model import make_model
from processor.processor import _pose_to_device


COCO_KP_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]

# 6-body-part grouping (matches skeleton_gcn.BODY_PART_GROUPS)
BODY_PART_GROUPS = [
    ("head", [0, 1, 2, 3, 4]),
    ("torso", [5, 6, 11, 12]),
    ("l_arm", [5, 7, 9]),
    ("r_arm", [6, 8, 10]),
    ("l_leg", [11, 13, 15]),
    ("r_leg", [12, 14, 16]),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file",
                   default="configs/market/pose_psg_lgpa_gcn_base.yml")
    p.add_argument("--weight",
                   default="log/market1501/exp260b_base_gcn512_2stage/"
                           "transformer_120.pth")
    p.add_argument("--dataset-root", default="data/occluded_reid")
    p.add_argument("--lo-thr", type=float, default=0.2,
                   help="score <= lo_thr -> occluded/low-vis token")
    p.add_argument("--hi-thr", type=float, default=0.7,
                   help="score >= hi_thr -> high-vis token")
    p.add_argument("--min-count", type=int, default=50,
                   help="min tokens per group per kp to report a stat")
    p.add_argument("--max-batches", type=int, default=0,
                   help="0 = all batches")
    p.add_argument("--out-json", default="scripts/vcnorm_probe_result.json")
    p.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return p.parse_args()


def build_loader(cfg, dataset_root):
    dataset = OccludedREID(dataset_dir=dataset_root)
    pose_root = os.path.join(dataset_root, "pose_data")
    hm_size = tuple(cfg.MODEL.POSE_HEATMAP_SIZE) \
        if hasattr(cfg.MODEL, "POSE_HEATMAP_SIZE") else None
    common = dict(
        img_size=tuple(cfg.INPUT.SIZE_TEST), is_train=False,
        pixel_mean=cfg.INPUT.PIXEL_MEAN, pixel_std=cfg.INPUT.PIXEL_STD,
        heatmap_size=hm_size,
    )
    q = PoseImageDataset(dataset.query,
                         pose_dir=os.path.join(pose_root, "query"), **common)
    g = PoseImageDataset(dataset.gallery,
                         pose_dir=os.path.join(pose_root, "gallery"), **common)
    val_set = ConcatDataset([q, g])
    loader = DataLoader(val_set, batch_size=cfg.TEST.IMS_PER_BATCH,
                        shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
                        collate_fn=pose_val_collate_fn)
    return dataset, loader


# ---------------------------------------------------------------------------
# distribution-distance helpers (per-channel diagonal-Gaussian, closed form)
# ---------------------------------------------------------------------------
def gaussian_kl_sym(mu_p, var_p, mu_q, var_q, eps=1e-6):
    """Symmetric diagonal-Gaussian KL, summed over channels.

    KL(P||Q) per channel = 0.5*(log(vq/vp) + (vp + (mu_p-mu_q)^2)/vq - 1)
    """
    var_p = np.maximum(var_p, eps)
    var_q = np.maximum(var_q, eps)
    d2 = (mu_p - mu_q) ** 2
    kl_pq = 0.5 * (np.log(var_q / var_p) + (var_p + d2) / var_q - 1.0)
    kl_qp = 0.5 * (np.log(var_p / var_q) + (var_q + d2) / var_p - 1.0)
    return float(np.sum(0.5 * (kl_pq + kl_qp)))


def gaussian_w2(mu_p, var_p, mu_q, var_q, eps=1e-6):
    """2-Wasserstein^2 for diagonal Gaussians, summed over channels.

    W2^2 = ||mu_p-mu_q||^2 + sum( (sqrt(vp)-sqrt(vq))^2 )
    Return sqrt for an interpretable distance.
    """
    var_p = np.maximum(var_p, eps)
    var_q = np.maximum(var_q, eps)
    mean_term = np.sum((mu_p - mu_q) ** 2)
    cov_term = np.sum((np.sqrt(var_p) - np.sqrt(var_q)) ** 2)
    return float(np.sqrt(max(mean_term + cov_term, 0.0)))


def lda_auc(X_hi, X_lo, eps=1e-6):
    """Fisher-LDA direction + ROC AUC of 1-D projection (separability proxy).

    Honest separability under the SAME diagonal-Gaussian assumption VC-Norm
    would exploit: w = (var_hi+var_lo)^{-1} (mu_hi - mu_lo).
    AUC computed on a held-out half to avoid trivial overfit.
    """
    n_hi, n_lo = X_hi.shape[0], X_lo.shape[0]
    if n_hi < 10 or n_lo < 10:
        return float("nan")
    rng = np.random.default_rng(0)
    hi_perm = rng.permutation(n_hi)
    lo_perm = rng.permutation(n_lo)
    hi_tr, hi_te = hi_perm[:n_hi // 2], hi_perm[n_hi // 2:]
    lo_tr, lo_te = lo_perm[:n_lo // 2], lo_perm[n_lo // 2:]

    mu_hi = X_hi[hi_tr].mean(0)
    mu_lo = X_lo[lo_tr].mean(0)
    var_hi = X_hi[hi_tr].var(0) + eps
    var_lo = X_lo[lo_tr].var(0) + eps
    w = (mu_hi - mu_lo) / (var_hi + var_lo)

    s_hi = X_hi[hi_te] @ w
    s_lo = X_lo[lo_te] @ w
    scores = np.concatenate([s_hi, s_lo])
    labels = np.concatenate([np.ones(len(s_hi)), np.zeros(len(s_lo))])
    # rank-based AUC
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(max(auc, 1.0 - auc))  # symmetric: separability magnitude


def main():
    args = parse_args()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    # force GCN part path active at test time so skeleton_head runs
    cfg.MODEL.POSE_TEST_FEAT = "equal_concat"
    cfg.TEST.WEIGHT = args.weight
    cfg.freeze()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", cfg.MODEL.DEVICE_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset, loader = build_loader(cfg, args.dataset_root)
    num_pids = len({pid for _, pid, _, _ in dataset.query + dataset.gallery})
    print(f"[probe] Occluded-ReID query={len(dataset.query)} "
          f"gallery={len(dataset.gallery)} ids={num_pids}", flush=True)

    model = make_model(cfg, num_class=max(num_pids, 1), camera_num=2,
                       view_num=1, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
    model.load_param(cfg.TEST.WEIGHT)
    model.to(device).eval()
    model.pose_test_feat = "equal_concat"

    # ----- hook skeleton_head to grab post-GCN per-kp tokens (aux_data) -----
    if not hasattr(model, "skeleton_head"):
        raise RuntimeError("model has no skeleton_head; exp260b should have GCN")

    captured = {}

    def hook(module, inputs, output):
        # SkeletonGCNHead.forward returns (cls_or_None, feats, aux_data)
        aux = output[2] if isinstance(output, tuple) and len(output) >= 3 else None
        if isinstance(aux, dict) and "kp_feats" in aux:
            captured["post_gcn"] = aux["kp_feats"].detach()
        # raw pre-GCN tokens: re-sample inside the same module from its input
        feat_map = inputs[0]
        pose_dict = inputs[1]
        with torch.no_grad():
            kp_feats, kp_scores = module._sample_keypoint_features(
                feat_map,
                pose_dict["keypoints"],
                pose_dict["scores"],
                pose_dict["person_mask"],
            )
        captured["pre_gcn"] = kp_feats.detach()
        captured["scores"] = kp_scores.detach()

    handle = model.skeleton_head.register_forward_hook(hook)

    # accumulators: per-keypoint, per-vis-group online mean/M2 (Welford) + bins
    K = 17
    C = None
    # store raw tokens for AUC only on a capped subsample to bound memory
    cap_per_group = 4000
    # raw_store[which][kp] = {'hi':[arr], 'lo':[arr], 'n_hi':int, 'n_lo':int}
    raw_store = {w: [{"hi": [], "lo": [], "n_hi": 0, "n_lo": 0}
                     for _ in range(K)] for w in ("pre_gcn", "post_gcn")}

    score_hist = np.zeros(11, dtype=np.int64)  # 0.0..1.0 in 0.1 bins
    n_tokens_total = 0

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if args.max_batches and bi >= args.max_batches:
                break
            img, pid, camid, camids, target_view, _, pose_dict = batch
            pose_dict = _pose_to_device(pose_dict, device)
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)

            captured.clear()
            _ = model(img, cam_label=camids, view_label=target_view,
                      pose_dict=pose_dict)

            if "pre_gcn" not in captured:
                raise RuntimeError("hook did not capture tokens; check path")

            scores = captured["scores"].float().cpu().numpy()  # (B,17)
            n_tokens_total += scores.size
            # score histogram
            hb = np.clip((scores.reshape(-1) * 10).astype(int), 0, 10)
            for b in hb:
                score_hist[b] += 1

            for which in ("pre_gcn", "post_gcn"):
                if which not in captured:
                    continue
                toks = captured[which].float().cpu().numpy()  # (B,17,C)
                if C is None:
                    C = toks.shape[-1]
                for k in range(K):
                    sc_k = scores[:, k]
                    hi_mask = sc_k >= args.hi_thr
                    lo_mask = sc_k <= args.lo_thr
                    store = raw_store[which][k]
                    if hi_mask.any():
                        arr = toks[hi_mask, k, :]
                        store["n_hi"] += arr.shape[0]
                        if store["n_hi"] - arr.shape[0] < cap_per_group:
                            take = min(arr.shape[0],
                                       cap_per_group - (store["n_hi"] - arr.shape[0]))
                            store["hi"].append(arr[:take])
                    if lo_mask.any():
                        arr = toks[lo_mask, k, :]
                        store["n_lo"] += arr.shape[0]
                        if store["n_lo"] - arr.shape[0] < cap_per_group:
                            take = min(arr.shape[0],
                                       cap_per_group - (store["n_lo"] - arr.shape[0]))
                            store["lo"].append(arr[:take])

            if bi % 10 == 0:
                print(f"[probe] batch {bi} tokens={n_tokens_total}", flush=True)

    handle.remove()

    print(f"\n[probe] feature dim C={C}", flush=True)
    print(f"[probe] score histogram (0.0..1.0 bins): "
          f"{score_hist.tolist()} (total={n_tokens_total})", flush=True)

    results = {
        "config": {
            "weight": args.weight, "lo_thr": args.lo_thr,
            "hi_thr": args.hi_thr, "min_count": args.min_count,
            "feat_dim": C, "n_tokens": int(n_tokens_total),
            "score_hist": score_hist.tolist(),
        },
        "per_kp": {},
        "per_part": {},
    }

    def summarize(which):
        rows = []
        for k in range(K):
            store = raw_store[which][k]
            n_hi, n_lo = store["n_hi"], store["n_lo"]
            if (n_hi < args.min_count or n_lo < args.min_count
                    or not store["hi"] or not store["lo"]):
                rows.append({
                    "kp": k, "name": COCO_KP_NAMES[k],
                    "n_hi": n_hi, "n_lo": n_lo, "skipped": True,
                })
                continue
            X_hi = np.concatenate(store["hi"], 0)
            X_lo = np.concatenate(store["lo"], 0)
            mu_hi, var_hi = X_hi.mean(0), X_hi.var(0)
            mu_lo, var_lo = X_lo.mean(0), X_lo.var(0)
            kl = gaussian_kl_sym(mu_hi, var_hi, mu_lo, var_lo)
            w2 = gaussian_w2(mu_hi, var_hi, mu_lo, var_lo)
            mean_l2 = float(np.linalg.norm(mu_hi - mu_lo))
            pooled_std = float(np.sqrt(np.mean(0.5 * (var_hi + var_lo))))
            mean_l2_norm = mean_l2 / (pooled_std * np.sqrt(len(mu_hi)) + 1e-9)
            auc = lda_auc(X_hi, X_lo)
            rows.append({
                "kp": k, "name": COCO_KP_NAMES[k],
                "n_hi": int(n_hi), "n_lo": int(n_lo),
                "kl_sym": kl, "w2": w2, "mean_l2": mean_l2,
                "mean_l2_norm": mean_l2_norm, "pooled_std": pooled_std,
                "lda_auc": auc, "skipped": False,
            })
        return rows

    print("\n" + "=" * 92)
    for which in ("pre_gcn", "post_gcn"):
        rows = summarize(which)
        results["per_kp"][which] = rows
        print(f"\n### {which.upper()} per-keypoint high-vis vs low-vis "
              f"(hi>={args.hi_thr}, lo<={args.lo_thr})")
        print(f"{'kp':<11} {'n_hi':>6} {'n_lo':>6} {'KL_sym':>10} "
              f"{'W2':>8} {'mean_l2':>8} {'meanZ':>7} {'LDA_AUC':>8}")
        vals_kl, vals_auc = [], []
        for r in rows:
            if r.get("skipped"):
                print(f"{r['name']:<11} {r['n_hi']:>6} {r['n_lo']:>6}  "
                      f"-- skipped (insufficient count) --")
                continue
            print(f"{r['name']:<11} {r['n_hi']:>6} {r['n_lo']:>6} "
                  f"{r['kl_sym']:>10.3f} {r['w2']:>8.3f} {r['mean_l2']:>8.3f} "
                  f"{r['mean_l2_norm']:>7.3f} {r['lda_auc']:>8.3f}")
            vals_kl.append(r["kl_sym"])
            vals_auc.append(r["lda_auc"])
        if vals_kl:
            print(f"{'-- median':<11} {'':>6} {'':>6} "
                  f"{np.median(vals_kl):>10.3f} {'':>8} {'':>8} {'':>7} "
                  f"{np.nanmedian(vals_auc):>8.3f}")
            results["per_kp"][which + "_summary"] = {
                "median_kl": float(np.median(vals_kl)),
                "mean_kl": float(np.mean(vals_kl)),
                "max_kl": float(np.max(vals_kl)),
                "median_auc": float(np.nanmedian(vals_auc)),
                "mean_auc": float(np.nanmean(vals_auc)),
                "max_auc": float(np.nanmax(vals_auc)),
            }

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[probe] wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
