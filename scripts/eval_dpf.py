#!/usr/bin/env python
"""
DPF: Distributional Part Features — Probabilistic Matching Evaluation

Test-time matching using per-keypoint feature VARIANCE as reliability
weights. Instead of confidence-score weighting (standard), uses
inverse-variance (precision) to weight per-keypoint comparisons.

Usage:
    python scripts/eval_dpf.py --config configs/occluded_duke/pose_psg_gcn_paa_dpf.yml \
        --weight log/occluded_duke/exp095_dpf/transformer_120.pth
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import cfg
from datasets import make_dataloader
from model import make_model
from processor.processor import _pose_to_device
from utils.metrics import eval_func


def extract_dpf_features(cfg_obj, model, val_loader, device='cuda'):
    """Extract global features, per-keypoint features, AND per-keypoint variances."""
    model.eval()
    use_pose = cfg_obj.MODEL.POSE_ENABLED

    all_global_feats = []
    all_kp_feats = []      # per-keypoint means (B, 17, D)
    all_kp_vars = []       # per-keypoint variances (B, 17, D) — DPF only
    all_kp_weights = []    # per-keypoint weights (B, 17)
    all_pids = []
    all_camids = []

    with torch.no_grad():
        for batch_data in val_loader:
            if use_pose:
                img, pid, camid, camids, target_view, imgpath, pose_dict = batch_data
                pose_dict = _pose_to_device(pose_dict, device)
            else:
                raise ValueError("DPF requires pose-enabled model")

            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)

            # Get model output
            test_feat, featmaps = model(img, cam_label=camids,
                                         view_label=target_view,
                                         pose_dict=pose_dict)

            # Extract global feat
            if isinstance(test_feat, dict):
                global_feat = test_feat['global_feat']
            else:
                global_feat = test_feat

            # Re-run skeleton head to get kp_feats + kp_vars
            _m = model.module if hasattr(model, 'module') else model
            if hasattr(_m, 'skeleton_head'):
                _, gcn_feats, aux_data = _m.skeleton_head(
                    featmaps[-1], pose_dict, return_cls=False)
                kp_feats_batch = aux_data.get('kp_feats')     # (B, 17, D)
                kp_vars_batch = aux_data.get('kp_vars')        # (B, 17, D) or None
                kp_weights_batch = aux_data.get('kp_weights')  # (B, 17)
            else:
                kp_feats_batch = None
                kp_vars_batch = None
                kp_weights_batch = None

            all_global_feats.append(global_feat.cpu())
            if kp_feats_batch is not None:
                all_kp_feats.append(kp_feats_batch.cpu())
                all_kp_weights.append(kp_weights_batch.cpu())
            if kp_vars_batch is not None:
                all_kp_vars.append(kp_vars_batch.cpu())
            all_pids.extend(np.asarray(pid))
            all_camids.extend(np.asarray(camid))

    global_feats = torch.cat(all_global_feats, dim=0)
    kp_feats = torch.cat(all_kp_feats, dim=0) if all_kp_feats else None
    kp_vars = torch.cat(all_kp_vars, dim=0) if all_kp_vars else None
    kp_weights = torch.cat(all_kp_weights, dim=0) if all_kp_weights else None
    pids = np.asarray(all_pids)
    camids = np.asarray(all_camids)

    return global_feats, kp_feats, kp_vars, kp_weights, pids, camids


def compute_precision_kp_distance(kp_feats_q, kp_feats_g, kp_vars_q, kp_vars_g):
    """Compute DPF precision-weighted per-keypoint distance.

    For each (query, gallery) pair and each keypoint k:
    - Compute cosine distance between keypoint features
    - Weight by joint precision: 1 / (var_q_k + var_g_k + eps)
    - Higher precision (lower variance) = more reliable comparison

    Args:
        kp_feats_q: (N_q, 17, D) query keypoint features
        kp_feats_g: (N_g, 17, D) gallery keypoint features
        kp_vars_q: (N_q, 17, D) query per-keypoint variances
        kp_vars_g: (N_g, 17, D) gallery per-keypoint variances

    Returns:
        distmat: (N_q, N_g) precision-weighted distance matrix
    """
    # Normalize features for cosine similarity
    kp_q_norm = F.normalize(kp_feats_q, dim=-1)  # (N_q, 17, D)
    kp_g_norm = F.normalize(kp_feats_g, dim=-1)  # (N_g, 17, D)

    # Per-keypoint cosine similarity: (N_q, N_g, 17)
    sim = torch.einsum('qkd,gkd->qgk', kp_q_norm, kp_g_norm)

    # Compute precision per keypoint pair: scalar variance per kp
    var_q = kp_vars_q.mean(dim=-1)  # (N_q, 17) — mean across feature dims
    var_g = kp_vars_g.mean(dim=-1)  # (N_g, 17)

    # Joint precision: 1 / (var_q + var_g + eps)
    # Shape: (N_q, 1, 17) + (1, N_g, 17) → (N_q, N_g, 17) via broadcast
    joint_var = var_q.unsqueeze(1) + var_g.unsqueeze(0)  # (N_q, N_g, 17)
    precision = 1.0 / (joint_var + 1e-3)
    precision = precision.clamp(max=1e3)  # prevent domination

    # Precision-weighted average similarity
    weighted_sim = (sim * precision).sum(dim=2) / precision.sum(dim=2).clamp(min=1e-6)

    # Distance = 1 - similarity
    return 1.0 - weighted_sim


def compute_confidence_kp_distance(kp_feats_q, kp_feats_g, weights_q, weights_g):
    """Standard confidence-weighted per-keypoint distance (for comparison)."""
    kp_q_norm = F.normalize(kp_feats_q, dim=-1)
    kp_g_norm = F.normalize(kp_feats_g, dim=-1)

    sim = torch.einsum('qkd,gkd->qgk', kp_q_norm, kp_g_norm)

    # Use min(w_q, w_g) as common visibility weight
    w_q = weights_q.unsqueeze(1)  # (N_q, 1, 17)
    w_g = weights_g.unsqueeze(0)  # (1, N_g, 17)
    common_w = torch.min(w_q, w_g)  # (N_q, N_g, 17)

    weighted_sim = (sim * common_w).sum(dim=2) / common_w.sum(dim=2).clamp(min=1e-6)
    return 1.0 - weighted_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--weight', required=True)
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for kp distance vs global distance')
    args = parser.parse_args()

    cfg.defrost()
    cfg.merge_from_file(args.config)
    cfg.TEST.WEIGHT = args.weight
    cfg.freeze()

    # Build dataloader and model
    (train_loader, _, val_loader,
     num_query, num_classes, camera_num, view_num) = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num,
                       view_num=view_num, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
    model.load_param(args.weight)
    model = model.cuda()

    print("Extracting features...")
    global_feats, kp_feats, kp_vars, kp_weights, pids, camids = \
        extract_dpf_features(cfg, model, val_loader)

    # Split query/gallery
    qf_global = F.normalize(global_feats[:num_query], dim=1)
    gf_global = F.normalize(global_feats[num_query:], dim=1)
    qf_kp = kp_feats[:num_query] if kp_feats is not None else None
    gf_kp = kp_feats[num_query:] if kp_feats is not None else None
    qw = kp_weights[:num_query] if kp_weights is not None else None
    gw = kp_weights[num_query:] if kp_weights is not None else None
    q_pids = pids[:num_query]
    g_pids = pids[num_query:]
    q_camids = camids[:num_query]
    g_camids = camids[num_query:]

    print(f"Query: {qf_global.shape[0]}, Gallery: {gf_global.shape[0]}")
    if kp_vars is not None:
        qv = kp_vars[:num_query]
        gv = kp_vars[num_query:]
        print(f"DPF variance available: {qv.shape}")
        # Print variance statistics
        mean_var = qv.mean(dim=-1)  # (N_q, 17)
        print(f"  Query mean variance per kp: min={mean_var.min():.4f}, "
              f"max={mean_var.max():.4f}, mean={mean_var.mean():.4f}")
    else:
        qv = gv = None
        print("No DPF variance available (non-DPF model)")

    # Step 1: Global distance (baseline)
    print("\n=== Step 1: Global cosine distance ===")
    distmat_global = (1 - torch.mm(qf_global, gf_global.t())).numpy()
    cmc, mAP = eval_func(distmat_global, q_pids, g_pids, q_camids, g_camids)
    print(f"  Global only: mAP={mAP*100:.1f}%, R1={cmc[0]*100:.1f}%")

    if qf_kp is None:
        print("No keypoint features, stopping.")
        return

    # Step 2: Confidence-weighted kp distance (standard)
    print("\n=== Step 2: Standard confidence-weighted kp matching ===")
    distmat_conf = compute_confidence_kp_distance(qf_kp, gf_kp, qw, gw).numpy()
    for alpha in [0.3, 0.5, 0.7]:
        dm = (1 - alpha) * distmat_global + alpha * distmat_conf
        cmc_a, mAP_a = eval_func(dm, q_pids, g_pids, q_camids, g_camids)
        print(f"  Conf-weighted (α={alpha}): mAP={mAP_a*100:.1f}%, R1={cmc_a[0]*100:.1f}%")

    # Step 3: DPF precision-weighted kp distance (our innovation)
    if qv is not None:
        print("\n=== Step 3: DPF precision-weighted kp matching ===")
        distmat_prec = compute_precision_kp_distance(qf_kp, gf_kp, qv, gv).numpy()
        for alpha in [0.3, 0.5, 0.7]:
            dm = (1 - alpha) * distmat_global + alpha * distmat_prec
            cmc_a, mAP_a = eval_func(dm, q_pids, g_pids, q_camids, g_camids)
            print(f"  Precision-weighted (α={alpha}): mAP={mAP_a*100:.1f}%, R1={cmc_a[0]*100:.1f}%")

        # Step 4: Compare precision vs confidence weighting
        print("\n=== Comparison: Precision vs Confidence weighting ===")
        best_alpha_conf = 0.5
        best_alpha_prec = 0.5
        dm_conf = (1 - best_alpha_conf) * distmat_global + best_alpha_conf * distmat_conf
        dm_prec = (1 - best_alpha_prec) * distmat_global + best_alpha_prec * distmat_prec
        cmc_conf, mAP_conf = eval_func(dm_conf, q_pids, g_pids, q_camids, g_camids)
        cmc_prec, mAP_prec = eval_func(dm_prec, q_pids, g_pids, q_camids, g_camids)
        print(f"  Confidence: mAP={mAP_conf*100:.1f}%, R1={cmc_conf[0]*100:.1f}%")
        print(f"  Precision:  mAP={mAP_prec*100:.1f}%, R1={cmc_prec[0]*100:.1f}%")
        delta_mAP = (mAP_prec - mAP_conf) * 100
        delta_R1 = (cmc_prec[0] - cmc_conf[0]) * 100
        print(f"  Delta:      mAP={delta_mAP:+.1f}%, R1={delta_R1:+.1f}%")


if __name__ == '__main__':
    main()
