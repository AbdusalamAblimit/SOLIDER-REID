"""Partial Optimal Transport (POT) distance evaluation for Occluded ReID.

Compares multiple test-time matching strategies on the same checkpoint:
1. Cosine on equal_concat (baseline)
2. MaxSim hybrid (current best)
3. POT hybrid with visibility-weighted mass
4. Visibility-weighted part cosine (simple baseline)

Usage:
    python scripts/eval_pot.py \
        --config_file configs/occluded_duke/pose_psg_lgpa_detach.yml \
        --weight log/occluded_duke/exp246b_lgpa_gcn/swin_tiny_120.pth \
        MODEL.POSE_LGPA True MODEL.POSE_LGPA_DETACH True MODEL.POSE_SKELETON_GCN True \
        MODEL.POSE_TEST_FEAT maxsim_hybrid
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg
from datasets.make_dataloader import make_dataloader
from model.make_model import make_model
from utils.metrics import eval_func


def extract_structured_features(model, loader, device='cuda'):
    """Extract global features, per-part features, and visibility weights.

    Returns:
        global_feats: (N, C) L2-normalized global features
        part_feats: (N, K, C) L2-normalized per-part features (K=5 for LGPA)
        vis_weights: (N, K) per-part visibility weights
        pids: (N,) person IDs
        camids: (N,) camera IDs
    """
    model.eval()
    all_global, all_parts, all_vis = [], [], []
    all_pids, all_camids = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            if len(batch) == 7:
                imgs, pid, camid, _, _, _, pose_dict = batch
            else:
                imgs, pid, camid = batch[0], batch[1], batch[2]
                pose_dict = batch[-1] if isinstance(batch[-1], dict) else None

            img = imgs.to(device)
            if pose_dict and isinstance(pose_dict, dict):
                pose_dict = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in pose_dict.items()
                }

            feat, _ = model(img, pose_dict=pose_dict)

            if isinstance(feat, dict):
                # maxsim_hybrid mode: structured features
                g = F.normalize(feat['global_feat'], p=2, dim=1)
                kp = F.normalize(feat['kp_feats'], p=2, dim=2)
                w = feat['kp_weights']
                all_global.append(g.cpu())
                all_parts.append(kp.cpu())
                all_vis.append(w.cpu())
            elif isinstance(feat, torch.Tensor):
                # equal_concat mode: split into global + parts
                C = 768
                n_feats = feat.shape[1] // C
                g = F.normalize(feat[:, :C], p=2, dim=1)
                all_global.append(g.cpu())
                if n_feats > 1:
                    # Parts are indices 1..N (skip global at 0)
                    # For LGPA+GCN: [global, lgpa_pooled, lgpa_p1..p5, gcn_pooled]
                    # Use lgpa parts (indices 2..6) if available
                    parts = feat[:, C:].reshape(-1, n_feats - 1, C)
                    parts = F.normalize(parts, p=2, dim=2)
                    all_parts.append(parts.cpu())
                    # No visibility info in concat mode — use uniform
                    all_vis.append(torch.ones(img.shape[0], n_feats - 1))
                else:
                    all_parts.append(None)
                    all_vis.append(None)

            all_pids.append(pid if isinstance(pid, torch.Tensor) else torch.tensor(pid))
            all_camids.append(camid if isinstance(camid, torch.Tensor) else torch.tensor(camid))

    global_feats = torch.cat(all_global)
    pids = torch.cat(all_pids).numpy()
    camids = torch.cat(all_camids).numpy()

    if all_parts[0] is not None:
        part_feats = torch.cat(all_parts)
        vis_weights = torch.cat(all_vis)
    else:
        part_feats = None
        vis_weights = None

    return global_feats, part_feats, vis_weights, pids, camids


def compute_cosine_distmat(qf, gf):
    """Standard cosine distance matrix."""
    return (1 - torch.mm(qf, gf.t())).numpy()


def compute_maxsim_distmat(q_global, g_global, q_parts, g_parts, q_vis, g_vis,
                           global_weight=1.0):
    """MaxSim hybrid distance: global cosine + max per-part cosine."""
    nq, ng = q_global.shape[0], g_global.shape[0]
    K = q_parts.shape[1]

    # Global distance
    gd = 1 - torch.mm(q_global, g_global.t())  # (nq, ng)

    # Part MaxSim: for each query part, find best-matching gallery part
    # Batched computation for efficiency
    distmat = np.zeros((nq, ng), dtype=np.float32)
    batch_size = 128

    for qi in tqdm(range(0, nq, batch_size), desc="MaxSim"):
        qe = min(qi + batch_size, nq)
        q_batch = q_parts[qi:qe]  # (bs, K, C)
        q_w = q_vis[qi:qe]  # (bs, K)

        for gi in range(0, ng, batch_size):
            ge = min(gi + batch_size, ng)
            # (bs, K, C) x (batch_g, K, C) -> (bs, batch_g, K_q, K_g)
            sim_block = torch.einsum('bkc,gpc->bgkp', q_batch, g_parts[gi:ge])
            # MaxSim: for each query part, take max over gallery parts
            maxsim = sim_block.max(dim=3)[0]  # (bs, batch_g, K_q)
            # Weight by query visibility
            weighted = (maxsim * q_w.unsqueeze(1)).sum(dim=2) / q_w.sum(dim=1, keepdim=True).clamp(min=1e-6)
            # Convert to distance
            distmat[qi:qe, gi:ge] = (1 - weighted).numpy()

    # Hybrid: blend global and part distances
    gd_np = gd.numpy()
    alpha = global_weight / (global_weight + 1.0)
    return alpha * gd_np + (1 - alpha) * distmat


def compute_pot_distmat(q_global, g_global, q_parts, g_parts, q_vis, g_vis,
                        mass_fraction=0.8, reg=0.05, global_weight=1.0, top_k=50):
    """POT hybrid: global cosine for initial ranking, then partial OT re-ranking.

    Args:
        mass_fraction: fraction of total mass to transport (handles occlusion)
                      'auto' = use min(sum(vis_q), sum(vis_g)) / max(...)
        reg: Sinkhorn entropic regularization
        global_weight: weight of global distance in hybrid
        top_k: re-rank only top-k candidates per query
    """
    try:
        import ot
    except ImportError:
        print("ERROR: POT library not installed. Run: pip install POT")
        return None

    nq, ng = q_global.shape[0], g_global.shape[0]
    K = q_parts.shape[1]

    # Global distance for initial ranking
    gd = (1 - torch.mm(q_global, g_global.t())).numpy()

    # POT re-ranking on top-k
    pot_dist = np.full((nq, ng), fill_value=999.0, dtype=np.float64)

    for i in tqdm(range(nq), desc="POT"):
        # Initial ranking by global distance
        di = gd[i].copy()

        # Exclude same-camera same-ID
        # (handled by eval_func, but we need valid top-k)
        top_idx = np.argsort(di)[:top_k]

        qi = q_parts[i].numpy().astype(np.float64)  # (K, C)
        vi = q_vis[i].numpy().astype(np.float64)  # (K,)

        for j_idx, j in enumerate(top_idx):
            gj = g_parts[j].numpy().astype(np.float64)  # (K, C)
            vj = g_vis[j].numpy().astype(np.float64)  # (K,)

            # Cost matrix: cosine distance between parts
            C = np.clip(1.0 - qi @ gj.T, 0, 2)  # (K, K)

            # Mass: proportional to visibility
            a = np.clip(vi, 1e-6, None)
            b = np.clip(vj, 1e-6, None)
            a = a / a.sum()
            b = b / b.sum()

            # Partial mass fraction
            if mass_fraction == 'auto':
                # Transport the minimum visible mass
                m = min(vi.sum(), vj.sum()) / max(vi.sum(), vj.sum(), 1e-6)
                m = max(0.3, min(m, 1.0))
            else:
                m = mass_fraction

            try:
                if m < 1.0:
                    # Partial OT: only transport fraction m of total mass
                    d = ot.partial.partial_wasserstein2(a, b, C, m=m)
                else:
                    # Full OT (balanced Sinkhorn)
                    T = ot.emd(a, b, C)
                    d = np.sum(T * C)
            except Exception:
                # Fallback to balanced OT
                T = ot.emd(a, b, C)
                d = np.sum(T * C)

            pot_dist[i, j] = d

    # Simple hybrid blending: re-rank top-k using POT distance
    # Strategy: for top-k candidates, blend global and POT distances directly
    # For non-top-k: keep global distance (they stay ranked below top-k)
    alpha = global_weight / (global_weight + 1.0)  # global weight fraction

    result = gd.copy()
    for i in range(nq):
        top_idx = np.argsort(gd[i])[:top_k]
        gd_top = gd[i, top_idx]
        pot_top = pot_dist[i, top_idx]

        # Scale POT to same range as global distance for fair blending
        pot_min, pot_max = pot_top.min(), pot_top.max()
        gd_min, gd_max = gd_top.min(), gd_top.max()
        if pot_max > pot_min and gd_max > gd_min:
            # Scale POT to match global distance range
            pot_scaled = (pot_top - pot_min) / (pot_max - pot_min) * (gd_max - gd_min) + gd_min
        else:
            pot_scaled = pot_top

        # Blend: keep original distance scale
        result[i, top_idx] = alpha * gd_top + (1 - alpha) * pot_scaled

    return result


def compute_vis_weighted_cosine(q_global, g_global, q_parts, g_parts, q_vis, g_vis,
                                global_weight=1.0):
    """Visibility-weighted per-part cosine distance (simple baseline).

    For each pair, compute: sum_k(w_qk * w_gk * cos(qk, gk)) / sum_k(w_qk * w_gk)
    """
    nq, ng = q_global.shape[0], g_global.shape[0]
    K = q_parts.shape[1]

    gd = (1 - torch.mm(q_global, g_global.t())).numpy()
    distmat = np.zeros((nq, ng), dtype=np.float32)

    batch_size = 128
    for qi in tqdm(range(0, nq, batch_size), desc="VisWeighted"):
        qe = min(qi + batch_size, nq)
        q_batch = q_parts[qi:qe]  # (bs, K, C)
        qw = q_vis[qi:qe]  # (bs, K)

        for gi in range(0, ng, batch_size):
            ge = min(gi + batch_size, ng)
            g_batch = g_parts[gi:ge]  # (bs_g, K, C)
            gw = g_vis[gi:ge]  # (bs_g, K)

            # Per-part cosine: (bs_q, K, C) x (bs_g, K, C) -> (bs_q, bs_g, K)
            sim = torch.einsum('bkc,gkc->bgk', q_batch, g_batch)

            # Visibility product weights: (bs_q, K) x (bs_g, K) -> (bs_q, bs_g, K)
            w = torch.einsum('bk,gk->bgk', qw, gw)
            w_sum = w.sum(dim=2).clamp(min=1e-6)

            # Weighted distance
            weighted_sim = (sim * w).sum(dim=2) / w_sum
            distmat[qi:qe, gi:ge] = (1 - weighted_sim).numpy()

    alpha = global_weight / (global_weight + 1.0)
    return alpha * gd + (1 - alpha) * distmat


def main():
    parser = argparse.ArgumentParser(description='POT distance evaluation')
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--weight', required=True)
    parser.add_argument('--top_k', type=int, default=50, help='Re-rank top-k for POT')
    parser.add_argument('--skip_pot', action='store_true', help='Skip slow POT, only run fast methods')
    args, remaining = parser.parse_known_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(remaining)
    cfg.TEST.WEIGHT = args.weight
    cfg.freeze()

    _, _, val_loader, num_query, num_classes, _, _ = make_dataloader(cfg)
    model = make_model(cfg, num_classes, 8, 0, cfg.MODEL.SEMANTIC_WEIGHT)
    model.load_param(args.weight)
    model.cuda()

    print("=" * 60)
    print("Extracting features...")
    print("=" * 60)
    global_feats, part_feats, vis_weights, pids, camids = extract_structured_features(model, val_loader)

    # Release GPU memory — all remaining computation is CPU-only
    del model
    torch.cuda.empty_cache()
    print("GPU memory released. All remaining computation is CPU-only.")

    qg = global_feats[:num_query]
    gg = global_feats[num_query:]
    qp = pids[:num_query]
    gp = pids[num_query:]
    qc = camids[:num_query]
    gc = camids[num_query:]

    K = part_feats.shape[1] if part_feats is not None else 0
    print(f"\nQuery: {qg.shape[0]}, Gallery: {gg.shape[0]}, Parts: {K}, Dim: {qg.shape[1]}")

    # 1. Global cosine baseline
    print("\n" + "=" * 60)
    print("1. Global Cosine Distance")
    print("=" * 60)
    distmat = compute_cosine_distmat(qg, gg)
    cmc, mAP = eval_func(distmat, qp, gp, qc, gc)
    print(f"   mAP={mAP * 100:.1f}%  R1={cmc[0] * 100:.1f}%  R5={cmc[4] * 100:.1f}%")

    if part_feats is None:
        print("No part features available. Exiting.")
        return

    q_parts = part_feats[:num_query]
    g_parts = part_feats[num_query:]
    q_vis = vis_weights[:num_query]
    g_vis = vis_weights[num_query:]

    # 2. Visibility-weighted per-part cosine
    print("\n" + "=" * 60)
    print("2. Visibility-Weighted Part Cosine (gw=1.0)")
    print("=" * 60)
    distmat_vw = compute_vis_weighted_cosine(qg, gg, q_parts, g_parts, q_vis, g_vis, global_weight=1.0)
    cmc_vw, mAP_vw = eval_func(distmat_vw, qp, gp, qc, gc)
    print(f"   mAP={mAP_vw * 100:.1f}%  R1={cmc_vw[0] * 100:.1f}%")

    # 3. MaxSim hybrid
    print("\n" + "=" * 60)
    print("3. MaxSim Hybrid (gw=1.0)")
    print("=" * 60)
    distmat_ms = compute_maxsim_distmat(qg, gg, q_parts, g_parts, q_vis, g_vis, global_weight=1.0)
    cmc_ms, mAP_ms = eval_func(distmat_ms, qp, gp, qc, gc)
    print(f"   mAP={mAP_ms * 100:.1f}%  R1={cmc_ms[0] * 100:.1f}%")

    if args.skip_pot:
        print("\nSkipping POT (--skip_pot). Done.")
        return

    # 4. POT hybrid with various mass fractions
    print("\n" + "=" * 60)
    print(f"4. Partial Optimal Transport (top_k={args.top_k})")
    print("=" * 60)

    for m in ['auto', 0.6, 0.8, 1.0]:
        for gw in [1.0]:
            m_str = m if isinstance(m, str) else f"{m:.1f}"
            print(f"\n   POT m={m_str}, gw={gw}")
            t0 = time.time()
            distmat_pot = compute_pot_distmat(
                qg, gg, q_parts, g_parts, q_vis, g_vis,
                mass_fraction=m, global_weight=gw, top_k=args.top_k)
            if distmat_pot is not None:
                cmc_pot, mAP_pot = eval_func(distmat_pot, qp, gp, qc, gc)
                elapsed = time.time() - t0
                print(f"   mAP={mAP_pot * 100:.1f}%  R1={cmc_pot[0] * 100:.1f}%  "
                      f"(Δ mAP{(mAP_pot - mAP) * 100:+.1f}%, time={elapsed:.0f}s)")

    # 5. MaxSim + POT combination: use MaxSim distance as base, POT re-ranks top-k
    print("\n" + "=" * 60)
    print(f"5. MaxSim + POT Combination (top_k={args.top_k})")
    print("=" * 60)

    for m in [0.6]:
        for pot_w in [0.3, 0.5, 0.7]:
            print(f"\n   MaxSim+POT m={m}, pot_weight={pot_w}")
            t0 = time.time()
            # Start from MaxSim distance
            result = distmat_ms.copy()
            import ot as ot_lib
            nq_c = q_parts.shape[0]
            K_c = q_parts.shape[1]
            for i in range(nq_c):
                top_idx = np.argsort(distmat_ms[i])[:args.top_k]
                qi = q_parts[i].numpy().astype(np.float64)
                vi = q_vis[i].numpy().astype(np.float64)
                ms_top = distmat_ms[i, top_idx]
                pot_top = np.zeros(args.top_k)
                for j_idx, j in enumerate(top_idx):
                    gj = g_parts[j].numpy().astype(np.float64)
                    vj = g_vis[j].numpy().astype(np.float64)
                    C = np.clip(1.0 - qi @ gj.T, 0, 2)
                    a = np.clip(vi, 1e-6, None); a = a / a.sum()
                    b = np.clip(vj, 1e-6, None); b = b / b.sum()
                    try:
                        pot_top[j_idx] = ot_lib.partial.partial_wasserstein2(a, b, C, m=m)
                    except Exception:
                        T = ot_lib.emd(a, b, C); pot_top[j_idx] = np.sum(T * C)
                # Scale POT to MaxSim range
                pt_min, pt_max = pot_top.min(), pot_top.max()
                ms_min, ms_max = ms_top.min(), ms_top.max()
                if pt_max > pt_min and ms_max > ms_min:
                    pot_scaled = (pot_top - pt_min) / (pt_max - pt_min) * (ms_max - ms_min) + ms_min
                else:
                    pot_scaled = pot_top
                result[i, top_idx] = (1 - pot_w) * ms_top + pot_w * pot_scaled
            cmc_mp, mAP_mp = eval_func(result, qp, gp, qc, gc)
            elapsed = time.time() - t0
            print(f"   mAP={mAP_mp * 100:.1f}%  R1={cmc_mp[0] * 100:.1f}%  "
                  f"(Δ vs MaxSim: mAP{(mAP_mp - mAP_ms) * 100:+.1f}%, R1{(cmc_mp[0] - cmc_ms[0]) * 100:+.1f}%, time={elapsed:.0f}s)")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Global cosine:       mAP={mAP * 100:.1f}%  R1={cmc[0] * 100:.1f}%")
    print(f"  Vis-weighted part:   mAP={mAP_vw * 100:.1f}%  R1={cmc_vw[0] * 100:.1f}%")
    print(f"  MaxSim hybrid:       mAP={mAP_ms * 100:.1f}%  R1={cmc_ms[0] * 100:.1f}%")


if __name__ == '__main__':
    main()
