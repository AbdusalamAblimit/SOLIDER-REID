#!/usr/bin/env python
"""
POT-Match: Optimal Transport Matching for Occluded Person ReID

Evaluate Sinkhorn OT distance between per-keypoint feature sets,
using pose confidence as transport mass weights.

Can be run on ANY existing checkpoint that has a skeleton GCN head.

Usage:
    python scripts/eval_pot.py --config configs/occluded_duke/pose_psg_gcn_paa.yml \
        --weight log/occluded_duke/exp066_paa/transformer_120.pth \
        --eps 0.1 --max_iter 20
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import cfg
from datasets import make_dataloader
from model import make_model
from processor.processor import _pose_to_device
from utils.metrics import eval_func


def sinkhorn_distance_batch(feat_q, feat_g, w_q, w_g, eps=0.1, max_iter=20):
    """Compute Sinkhorn OT distance for a batch of pairs.

    Args:
        feat_q: (N, K, D) query keypoint features
        feat_g: (N, K, D) gallery keypoint features
        w_q: (N, K) query weights
        w_g: (N, K) gallery weights

    Returns:
        dist: (N,) OT distances
    """
    # Normalize features
    fq = F.normalize(feat_q, dim=-1)
    fg = F.normalize(feat_g, dim=-1)

    # Cost: 1 - cosine similarity
    cost = 1.0 - torch.bmm(fq, fg.transpose(1, 2))  # (N, K, K)

    # Normalize weights to distributions
    mu = w_q.clamp(min=1e-8)
    mu = mu / mu.sum(dim=1, keepdim=True)
    nu = w_g.clamp(min=1e-8)
    nu = nu / nu.sum(dim=1, keepdim=True)

    log_mu = torch.log(mu)
    log_nu = torch.log(nu)
    M = -cost / eps

    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)

    for _ in range(max_iter):
        u = log_mu - torch.logsumexp(M + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(M + u.unsqueeze(2), dim=1)

    log_T = M + u.unsqueeze(2) + v.unsqueeze(1)
    T = torch.exp(log_T)
    dist = (T * cost).sum(dim=(1, 2))

    return dist


def compute_ot_distmat(qf_kp, gf_kp, qw, gw, eps=0.1, max_iter=20,
                       batch_size=256, device='cuda'):
    """Compute pairwise OT distance matrix between queries and gallery.

    Args:
        qf_kp: (N_q, K, D) query keypoint features
        gf_kp: (N_g, K, D) gallery keypoint features
        qw: (N_q, K) query weights
        gw: (N_g, K) gallery weights

    Returns:
        distmat: (N_q, N_g) distance matrix
    """
    N_q = qf_kp.shape[0]
    N_g = gf_kp.shape[0]
    distmat = np.zeros((N_q, N_g), dtype=np.float32)

    gf_kp_dev = gf_kp.to(device)
    gw_dev = gw.to(device)

    for qi in tqdm(range(N_q), desc='OT matching'):
        # Expand query to match all gallery
        q_feat = qf_kp[qi:qi+1].expand(N_g, -1, -1).to(device)  # (N_g, K, D)
        q_w = qw[qi:qi+1].expand(N_g, -1).to(device)  # (N_g, K)

        # Compute in batches to avoid OOM
        dists = []
        for start in range(0, N_g, batch_size):
            end = min(start + batch_size, N_g)
            d = sinkhorn_distance_batch(
                q_feat[start:end], gf_kp_dev[start:end],
                q_w[start:end], gw_dev[start:end],
                eps=eps, max_iter=max_iter)
            dists.append(d.cpu())

        distmat[qi] = torch.cat(dists).numpy()

    return distmat


def extract_kp_features(cfg_obj, model, val_loader, device='cuda'):
    """Extract global + per-keypoint features."""
    model.eval()
    all_global, all_kp, all_kp_w = [], [], []
    all_pids, all_camids = [], []

    with torch.no_grad():
        for batch_data in val_loader:
            img, pid, camid, camids, target_view, imgpath, pose_dict = batch_data
            pose_dict = _pose_to_device(pose_dict, device)
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)

            test_feat, featmaps = model(img, cam_label=camids,
                                         view_label=target_view,
                                         pose_dict=pose_dict)
            if isinstance(test_feat, dict):
                global_feat = test_feat['global_feat']
            else:
                global_feat = test_feat

            _m = model.module if hasattr(model, 'module') else model
            if hasattr(_m, 'skeleton_head'):
                _, _, aux_data = _m.skeleton_head(
                    featmaps[-1], pose_dict, return_cls=False)
                kp_feats = aux_data.get('kp_feats')
                kp_weights = aux_data.get('kp_weights')
            else:
                raise ValueError("Model has no skeleton_head")

            all_global.append(global_feat.cpu())
            all_kp.append(kp_feats.cpu())
            all_kp_w.append(kp_weights.cpu())
            all_pids.extend(np.asarray(pid))
            all_camids.extend(np.asarray(camid))

    return (torch.cat(all_global), torch.cat(all_kp), torch.cat(all_kp_w),
            np.asarray(all_pids), np.asarray(all_camids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--weight', required=True)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--max_iter', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()

    cfg.defrost()
    cfg.merge_from_file(args.config)
    cfg.TEST.WEIGHT = args.weight
    cfg.freeze()

    (train_loader, _, val_loader,
     num_query, num_classes, camera_num, view_num) = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num,
                       view_num=view_num, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
    model.load_param(args.weight)
    model = model.cuda()

    print("Extracting features...")
    global_feats, kp_feats, kp_weights, pids, camids = \
        extract_kp_features(cfg, model, val_loader)

    qf_g = F.normalize(global_feats[:num_query], dim=1)
    gf_g = F.normalize(global_feats[num_query:], dim=1)
    qf_kp = kp_feats[:num_query]
    gf_kp = kp_feats[num_query:]
    qw = kp_weights[:num_query]
    gw = kp_weights[num_query:]
    q_pids, g_pids = pids[:num_query], pids[num_query:]
    q_camids, g_camids = camids[:num_query], camids[num_query:]

    print(f"Q: {qf_g.shape[0]}, G: {gf_g.shape[0]}, KP: {qf_kp.shape}")

    # Baseline: global cosine
    print("\n=== Global cosine (baseline) ===")
    distmat_global = (1 - torch.mm(qf_g, gf_g.t())).numpy()
    cmc, mAP = eval_func(distmat_global, q_pids, g_pids, q_camids, g_camids)
    print(f"  mAP={mAP*100:.1f}%, R1={cmc[0]*100:.1f}%")

    # OT matching
    print(f"\n=== Sinkhorn OT matching (eps={args.eps}, iter={args.max_iter}) ===")
    distmat_ot = compute_ot_distmat(
        qf_kp, gf_kp, qw, gw,
        eps=args.eps, max_iter=args.max_iter)
    cmc_ot, mAP_ot = eval_func(distmat_ot, q_pids, g_pids, q_camids, g_camids)
    print(f"  OT-only: mAP={mAP_ot*100:.1f}%, R1={cmc_ot[0]*100:.1f}%")

    # Hybrid: global + OT
    print(f"\n=== Hybrid: global + OT ===")
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        dm = (1 - alpha) * distmat_global + alpha * distmat_ot
        cmc_h, mAP_h = eval_func(dm, q_pids, g_pids, q_camids, g_camids)
        marker = " ★" if mAP_h > mAP else ""
        print(f"  α={alpha}: mAP={mAP_h*100:.1f}%, R1={cmc_h[0]*100:.1f}%{marker}")


if __name__ == '__main__':
    main()
