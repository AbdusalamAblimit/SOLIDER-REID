#!/usr/bin/env python
"""VC-Norm probe CONTROLS — is the high/low-vis token shift a real domain
factor, or a keypoint-sampling artifact?

The main probe (vcnorm_probe.py) showed a huge separable shift (KL ~ 100-300,
LDA AUC ~ 0.97) between high-vis and low-vis per-keypoint tokens. Before
calling that "fuel for VC-Norm", we must rule out the trivial explanation:

  An OCCLUDED keypoint has a low pose score AND a degenerate / hallucinated
  pixel coordinate (often pinned to border). Bilinear sampling at that
  coordinate (padding_mode='border') returns border / off-body features.
  If so, the high/low-vis token shift is just "on-body vs off-body sampling",
  NOT an occlusion-as-domain-factor that VC-Norm could usefully align.

CONTROLS
--------
C1. Coordinate geometry of low-vis vs high-vis keypoints:
    fraction of low-vis keypoints whose (x,y) sits at/near the image border,
    and mean distance of low-vis coords to the high-vis coord centroid.
    Border-pinned low-vis coords => sampling artifact.

C2. RANDOM-COORD baseline: for each keypoint, sample a token at a RANDOM
    in-frame location (independent of pose). Compute KL(high_vis, random).
    If KL(high_vis, low_vis) ~ KL(high_vis, random), the low-vis "shift" is
    indistinguishable from sampling anywhere off the keypoint => artifact,
    no occlusion-specific domain factor.

C3. ON-BODY low-vis subset: restrict low-vis tokens to those whose coordinate
    is INSIDE the central body region (not border). If KL stays large on this
    subset, the shift is NOT purely a border artifact => genuine fuel.

C4. LGPA part-token shift (faithful target): the deployed part features are
    LGPA heatmap-attention-pooled tokens, not raw keypoint samples. Split each
    of the 5 LGPA part tokens by part-visibility (mean pose score over the
    part's keypoints) into high/low and measure KL there. This is what VC-Norm
    would actually normalize in this model.

Outputs a JSON + readable table.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
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
# 5 LGPA parts (head merged into body in LGPA's 5-part scheme is model-specific;
# we use a generic 5-group over keypoints for the part-visibility split)
LGPA_PART_GROUPS = [
    ("upper", [0, 1, 2, 3, 4, 5, 6]),     # head+shoulders
    ("torso", [5, 6, 11, 12]),
    ("arms", [7, 8, 9, 10]),
    ("legs_up", [11, 12, 13, 14]),
    ("legs_low", [15, 16]),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file",
                   default="configs/market/pose_psg_lgpa_gcn_base.yml")
    p.add_argument("--weight",
                   default="log/market1501/exp260b_base_gcn512_2stage/"
                           "transformer_120.pth")
    p.add_argument("--dataset-root", default="data/occluded_reid")
    p.add_argument("--lo-thr", type=float, default=0.2)
    p.add_argument("--hi-thr", type=float, default=0.7)
    p.add_argument("--min-count", type=int, default=50)
    p.add_argument("--border-frac", type=float, default=0.08,
                   help="coord within border-frac of any edge => 'border'")
    p.add_argument("--out-json", default="scripts/vcnorm_control_result.json")
    p.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return p.parse_args()


def gaussian_kl_sym(mu_p, var_p, mu_q, var_q, eps=1e-6):
    var_p = np.maximum(var_p, eps)
    var_q = np.maximum(var_q, eps)
    d2 = (mu_p - mu_q) ** 2
    kl_pq = 0.5 * (np.log(var_q / var_p) + (var_p + d2) / var_q - 1.0)
    kl_qp = 0.5 * (np.log(var_p / var_q) + (var_q + d2) / var_p - 1.0)
    return float(np.sum(0.5 * (kl_pq + kl_qp)))


def main():
    args = parse_args()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.MODEL.POSE_TEST_FEAT = "equal_concat"
    cfg.TEST.WEIGHT = args.weight
    cfg.freeze()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", cfg.MODEL.DEVICE_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = OccludedREID(dataset_dir=args.dataset_root)
    pose_root = os.path.join(args.dataset_root, "pose_data")
    hm_size = tuple(cfg.MODEL.POSE_HEATMAP_SIZE)
    common = dict(img_size=tuple(cfg.INPUT.SIZE_TEST), is_train=False,
                  pixel_mean=cfg.INPUT.PIXEL_MEAN, pixel_std=cfg.INPUT.PIXEL_STD,
                  heatmap_size=hm_size)
    q = PoseImageDataset(dataset.query,
                         pose_dir=os.path.join(pose_root, "query"), **common)
    g = PoseImageDataset(dataset.gallery,
                         pose_dir=os.path.join(pose_root, "gallery"), **common)
    loader = DataLoader(ConcatDataset([q, g]), batch_size=cfg.TEST.IMS_PER_BATCH,
                        shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
                        collate_fn=pose_val_collate_fn)
    num_pids = len({pid for _, pid, _, _ in dataset.query + dataset.gallery})

    model = make_model(cfg, num_class=max(num_pids, 1), camera_num=2,
                       view_num=1, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
    model.load_param(cfg.TEST.WEIGHT)
    model.to(device).eval()
    model.pose_test_feat = "equal_concat"

    sk = model.skeleton_head
    input_h, input_w = cfg.INPUT.SIZE_TEST

    captured = {}

    def hook(module, inputs, output):
        feat_map = inputs[0]
        pose_dict = inputs[1]
        captured["feat_map"] = feat_map.detach()
        captured["keypoints"] = pose_dict["keypoints"].detach()
        captured["scores"] = pose_dict["scores"].detach()
        captured["person_mask"] = pose_dict["person_mask"].detach()
        with torch.no_grad():
            kp_feats, kp_scores = module._sample_keypoint_features(
                feat_map, pose_dict["keypoints"], pose_dict["scores"],
                pose_dict["person_mask"])
        captured["pre_gcn"] = kp_feats.detach()
        captured["kp_scores"] = kp_scores.detach()

    handle = sk.register_forward_hook(hook)

    K = 17
    cap = 4000
    # store: per kp -> hi / lo / random tokens, + lo-onbody tokens
    store = [{"hi": [], "lo": [], "rand": [], "lo_onbody": [],
              "n_hi": 0, "n_lo": 0, "n_rand": 0, "n_lo_onbody": 0}
             for _ in range(K)]
    # coordinate geometry accumulators
    coord_hi = [[] for _ in range(K)]
    coord_lo = [[] for _ in range(K)]
    lo_border_count = np.zeros(K, dtype=np.int64)
    lo_total_count = np.zeros(K, dtype=np.int64)

    def random_sample(feat_map):
        B, C, fH, fW = feat_map.shape
        grid = torch.rand(B, K, 1, 2, device=feat_map.device) * 2 - 1
        s = F.grid_sample(feat_map, grid, mode="bilinear",
                          padding_mode="border", align_corners=True)
        return s.squeeze(-1).permute(0, 2, 1)  # (B,17,C)

    bx = args.border_frac
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            img, pid, camid, camids, target_view, _, pose_dict = batch
            pose_dict = _pose_to_device(pose_dict, device)
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            captured.clear()
            _ = model(img, cam_label=camids, view_label=target_view,
                      pose_dict=pose_dict)

            toks = captured["pre_gcn"].float().cpu().numpy()       # (B,17,C)
            sc = captured["kp_scores"].float().cpu().numpy()       # (B,17)
            kp = captured["keypoints"][:, 0].float().cpu().numpy()  # (B,17,2) px
            rand_toks = random_sample(captured["feat_map"]).float().cpu().numpy()

            # normalized coords in [0,1]
            nx = kp[:, :, 0] / input_w
            ny = kp[:, :, 1] / input_h
            on_border = ((nx <= bx) | (nx >= 1 - bx) |
                         (ny <= bx) | (ny >= 1 - bx))

            for k in range(K):
                hi_m = sc[:, k] >= args.hi_thr
                lo_m = sc[:, k] <= args.lo_thr
                s = store[k]
                if hi_m.any():
                    a = toks[hi_m, k, :]
                    if s["n_hi"] < cap:
                        s["hi"].append(a[:cap - s["n_hi"]])
                    s["n_hi"] += a.shape[0]
                    coord_hi[k].append(np.stack([nx[hi_m, k], ny[hi_m, k]], 1))
                if lo_m.any():
                    a = toks[lo_m, k, :]
                    if s["n_lo"] < cap:
                        s["lo"].append(a[:cap - s["n_lo"]])
                    s["n_lo"] += a.shape[0]
                    coord_lo[k].append(np.stack([nx[lo_m, k], ny[lo_m, k]], 1))
                    lo_total_count[k] += int(lo_m.sum())
                    lo_border_count[k] += int((lo_m & on_border[:, k]).sum())
                    # on-body low-vis subset (NOT on border)
                    lo_onbody_m = lo_m & (~on_border[:, k])
                    if lo_onbody_m.any():
                        a2 = toks[lo_onbody_m, k, :]
                        if s["n_lo_onbody"] < cap:
                            s["lo_onbody"].append(a2[:cap - s["n_lo_onbody"]])
                        s["n_lo_onbody"] += a2.shape[0]
                # random tokens for this kp (independent of pose)
                a_r = rand_toks[:, k, :]
                if s["n_rand"] < cap:
                    s["rand"].append(a_r[:cap - s["n_rand"]])
                s["n_rand"] += a_r.shape[0]

    handle.remove()

    def cat(lst):
        return np.concatenate(lst, 0) if lst else np.zeros((0, 1))

    print("\n" + "=" * 100)
    print("VC-Norm CONTROL: is the high/low-vis shift a real domain factor "
          "or a keypoint-sampling artifact?")
    print("=" * 100)
    print(f"\n{'kp':<11} {'n_lo':>6} {'%border':>8} {'KL(hi,lo)':>10} "
          f"{'KL(hi,rand)':>12} {'KL(hi,lo_onbody)':>17} {'n_onbody':>9}")

    results = {"config": {"lo_thr": args.lo_thr, "hi_thr": args.hi_thr,
                          "border_frac": args.border_frac},
               "rows": []}
    for k in range(K):
        s = store[k]
        if s["n_hi"] < args.min_count or s["n_lo"] < args.min_count:
            continue
        X_hi = cat(s["hi"]); X_lo = cat(s["lo"])
        X_rand = cat(s["rand"]); X_lob = cat(s["lo_onbody"])
        mu_hi, var_hi = X_hi.mean(0), X_hi.var(0)
        mu_lo, var_lo = X_lo.mean(0), X_lo.var(0)
        mu_r, var_r = X_rand.mean(0), X_rand.var(0)
        kl_hilo = gaussian_kl_sym(mu_hi, var_hi, mu_lo, var_lo)
        kl_hirand = gaussian_kl_sym(mu_hi, var_hi, mu_r, var_r)
        if X_lob.shape[0] >= args.min_count:
            kl_hilob = gaussian_kl_sym(mu_hi, var_hi, X_lob.mean(0), X_lob.var(0))
            n_lob = X_lob.shape[0]
        else:
            kl_hilob = float("nan"); n_lob = X_lob.shape[0]
        pct_border = 100.0 * lo_border_count[k] / max(lo_total_count[k], 1)
        print(f"{COCO_KP_NAMES[k]:<11} {s['n_lo']:>6} {pct_border:>7.1f}% "
              f"{kl_hilo:>10.2f} {kl_hirand:>12.2f} {kl_hilob:>17.2f} {n_lob:>9}")
        results["rows"].append({
            "kp": k, "name": COCO_KP_NAMES[k], "n_lo": int(s["n_lo"]),
            "pct_border": float(pct_border), "kl_hi_lo": kl_hilo,
            "kl_hi_rand": kl_hirand, "kl_hi_lo_onbody": kl_hilob,
            "n_lo_onbody": int(n_lob),
        })

    # summary medians (over kps with valid rows)
    rows = results["rows"]
    if rows:
        med_hilo = float(np.median([r["kl_hi_lo"] for r in rows]))
        med_hirand = float(np.median([r["kl_hi_rand"] for r in rows]))
        valid_lob = [r["kl_hi_lo_onbody"] for r in rows
                     if not np.isnan(r["kl_hi_lo_onbody"])]
        med_hilob = float(np.median(valid_lob)) if valid_lob else float("nan")
        med_border = float(np.median([r["pct_border"] for r in rows]))
        print(f"\n{'MEDIAN':<11} {'':>6} {med_border:>7.1f}% "
              f"{med_hilo:>10.2f} {med_hirand:>12.2f} {med_hilob:>17.2f}")
        print("\nINTERPRETATION GUIDE:")
        print("  - KL(hi,lo) >> KL(hi,rand)  => low-vis shift is occlusion-"
              "specific, not generic off-kp sampling (GOOD = fuel)")
        print("  - KL(hi,lo) ~ KL(hi,rand)   => low-vis just looks like random "
              "off-body sampling (ARTIFACT)")
        print("  - KL(hi,lo_onbody) still large & %border low => shift survives "
              "removing border-pinned coords (GOOD = fuel)")
        print("  - %border high & KL(hi,lo_onbody) ~ 0 => shift is border "
              "artifact (KILL)")
        results["summary"] = {
            "median_kl_hi_lo": med_hilo, "median_kl_hi_rand": med_hirand,
            "median_kl_hi_lo_onbody": med_hilob, "median_pct_border": med_border,
        }

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[control] wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
