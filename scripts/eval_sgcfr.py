#!/usr/bin/env python
"""
Skeleton-Guided Cross-Image Feature Recovery (SGCFR)

Test-time feature recovery: use gallery candidates' visible keypoint features
to fill in query's occluded keypoint features, guided by skeleton structure.

Usage:
    python scripts/eval_sgcfr.py --config configs/occluded_duke/pose_psg_gcn_paa.yml \
        --weight log/occluded_duke/exp066_paa/transformer_120.pth \
        --top_k 10 --vis_threshold 0.3
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


def extract_features_with_kp(cfg_obj, model, val_loader, device='cuda'):
    """Extract global features AND per-keypoint features."""
    model.eval()
    use_pose = cfg_obj.MODEL.POSE_ENABLED

    all_global_feats = []
    all_kp_feats = []
    all_kp_weights = []
    all_pids = []
    all_camids = []

    with torch.no_grad():
        for batch_data in val_loader:
            if use_pose:
                img, pid, camid, camids, target_view, imgpath, pose_dict = batch_data
                pose_dict = _pose_to_device(pose_dict, device)
            else:
                raise ValueError("SGCFR requires pose-enabled model")

            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)

            # Get model output
            test_feat, featmaps = model(img, cam_label=camids,
                                         view_label=target_view,
                                         pose_dict=pose_dict)

            # If test_feat is dict (cvk mode), extract global
            if isinstance(test_feat, dict):
                global_feat = test_feat['global_feat']
                kp_feats_batch = test_feat.get('kp_feats')
                kp_weights_batch = test_feat.get('kp_weights')
            else:
                # equal_concat mode: need to also get kp_feats separately
                global_feat = test_feat
                # Re-run skeleton head to get kp_feats
                _m = model.module if hasattr(model, 'module') else model
                if hasattr(_m, 'skeleton_head'):
                    _, gcn_feats, aux_data = _m.skeleton_head(
                        featmaps[-1], pose_dict, return_cls=False)
                    kp_feats_batch = aux_data.get('kp_feats')
                    kp_weights_batch = aux_data.get('kp_weights')
                else:
                    kp_feats_batch = None
                    kp_weights_batch = None

            all_global_feats.append(global_feat.cpu())
            if kp_feats_batch is not None:
                all_kp_feats.append(kp_feats_batch.cpu())
                all_kp_weights.append(kp_weights_batch.cpu())
            all_pids.extend(np.asarray(pid))
            all_camids.extend(np.asarray(camid))

    global_feats = torch.cat(all_global_feats, dim=0)
    kp_feats = torch.cat(all_kp_feats, dim=0) if all_kp_feats else None
    kp_weights = torch.cat(all_kp_weights, dim=0) if all_kp_weights else None
    pids = np.asarray(all_pids)
    camids = np.asarray(all_camids)

    return global_feats, kp_feats, kp_weights, pids, camids


def sgcfr_recover(query_kp_feats, query_kp_weights,
                   gallery_kp_feats, gallery_kp_weights,
                   initial_ranking, top_k=10, vis_threshold=0.3):
    """Recover query's occluded keypoint features using gallery candidates.

    Args:
        query_kp_feats: (N_q, 17, D)
        query_kp_weights: (N_q, 17)
        gallery_kp_feats: (N_g, 17, D)
        gallery_kp_weights: (N_g, 17)
        initial_ranking: (N_q, N_g) indices from initial L2 ranking
        top_k: number of candidates to use for recovery
        vis_threshold: below this, keypoint is considered occluded

    Returns:
        recovered_kp_feats: (N_q, 17, D) with occluded kp replaced
        recovered_weights: (N_q, 17) updated weights
    """
    N_q = query_kp_feats.shape[0]
    recovered_feats = query_kp_feats.clone()
    recovered_weights = query_kp_weights.clone()

    for qi in range(N_q):
        # Get top-K gallery candidates
        top_k_indices = initial_ranking[qi, :top_k]

        for kp in range(17):
            if query_kp_weights[qi, kp] >= vis_threshold:
                continue  # This keypoint is visible, no recovery needed

            # Find candidates where this keypoint is visible
            cand_vis = gallery_kp_weights[top_k_indices, kp]  # (top_k,)
            visible_mask = cand_vis >= vis_threshold

            if not visible_mask.any():
                continue  # No candidate has this kp visible

            # Weighted average of visible candidates' kp features
            visible_cands = top_k_indices[visible_mask]
            cand_feats = gallery_kp_feats[visible_cands, kp]  # (n_vis, D)
            cand_weights = gallery_kp_weights[visible_cands, kp]  # (n_vis,)

            # Weight by visibility score
            w = cand_weights / cand_weights.sum()
            recovered_feat = (cand_feats * w.unsqueeze(1)).sum(dim=0)

            recovered_feats[qi, kp] = recovered_feat
            recovered_weights[qi, kp] = cand_weights.mean()

    return recovered_feats, recovered_weights


def compute_kp_distance(kp_feats_q, kp_feats_g, weights_q, weights_g):
    """Compute CVK-style distance using keypoint features.

    For each query-gallery pair, compute weighted cosine distance
    only on commonly visible keypoints.
    """
    N_q, N_kp, D = kp_feats_q.shape
    N_g = kp_feats_g.shape[0]

    # Normalize features
    kp_q_norm = F.normalize(kp_feats_q, dim=-1)  # (N_q, 17, D)
    kp_g_norm = F.normalize(kp_feats_g, dim=-1)  # (N_g, 17, D)

    # Compute per-keypoint similarity: (N_q, N_g, 17)
    # Using einsum for efficiency
    sim = torch.einsum('qkd,gkd->qgk', kp_q_norm, kp_g_norm)

    # Common visibility mask: (N_q, N_g, 17)
    vis_q = (weights_q > 0.3).float()  # (N_q, 17)
    vis_g = (weights_g > 0.3).float()  # (N_g, 17)
    common_vis = torch.einsum('qi,gi->qgi', vis_q, vis_g)

    # Masked average similarity → distance
    masked_sim = (sim * common_vis).sum(dim=2)  # (N_q, N_g)
    vis_count = common_vis.sum(dim=2).clamp(min=1)  # (N_q, N_g)
    avg_sim = masked_sim / vis_count

    # Distance = 1 - similarity
    dist = 1.0 - avg_sim

    return dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--weight', required=True)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--vis_threshold', type=float, default=0.3)
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for recovered kp distance vs global distance')
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

    print(f"Extracting features...")
    global_feats, kp_feats, kp_weights, pids, camids = \
        extract_features_with_kp(cfg, model, val_loader)

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
    print(f"KP feats: {qf_kp.shape if qf_kp is not None else 'None'}")

    # Step 1: Initial ranking with global features (L2)
    print("Step 1: Initial L2 ranking...")
    distmat_global = torch.cdist(qf_global, gf_global, p=2).numpy()
    initial_ranking = np.argsort(distmat_global, axis=1)

    # Baseline eval
    cmc, mAP = eval_func(distmat_global, q_pids, g_pids, q_camids, g_camids)
    print(f"\nBaseline (global L2): mAP={mAP*100:.1f}%, R1={cmc[0]*100:.1f}%")

    if qf_kp is None:
        print("No keypoint features available, stopping.")
        return

    # Step 2: CVK matching without recovery
    print("\nStep 2: CVK matching (no recovery)...")
    distmat_kp = compute_kp_distance(qf_kp, gf_kp, qw, gw).numpy()
    distmat_hybrid = (1 - args.alpha) * distmat_global + args.alpha * distmat_kp
    cmc, mAP = eval_func(distmat_hybrid, q_pids, g_pids, q_camids, g_camids)
    print(f"CVK hybrid (alpha={args.alpha}): mAP={mAP*100:.1f}%, R1={cmc[0]*100:.1f}%")

    # Step 3: SGCFR recovery
    print(f"\nStep 3: SGCFR recovery (top_k={args.top_k}, threshold={args.vis_threshold})...")
    recovered_kp, recovered_w = sgcfr_recover(
        qf_kp, qw, gf_kp, gw,
        initial_ranking, top_k=args.top_k, vis_threshold=args.vis_threshold)

    # Step 4: Re-compute distance with recovered features
    distmat_recovered = compute_kp_distance(recovered_kp, gf_kp, recovered_w, gw).numpy()
    distmat_sgcfr = (1 - args.alpha) * distmat_global + args.alpha * distmat_recovered
    cmc, mAP = eval_func(distmat_sgcfr, q_pids, g_pids, q_camids, g_camids)
    print(f"SGCFR (alpha={args.alpha}): mAP={mAP*100:.1f}%, R1={cmc[0]*100:.1f}%")

    # Test different alphas
    print("\nAlpha sensitivity:")
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        dm = (1 - alpha) * distmat_global + alpha * distmat_recovered
        cmc_a, mAP_a = eval_func(dm, q_pids, g_pids, q_camids, g_camids)
        print(f"  alpha={alpha}: mAP={mAP_a*100:.1f}%, R1={cmc_a[0]*100:.1f}%")


if __name__ == '__main__':
    main()
