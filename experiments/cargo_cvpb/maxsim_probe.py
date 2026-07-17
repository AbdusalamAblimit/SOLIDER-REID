# encoding: utf-8
"""
CVPB kill-switch #1 -- zero-training local-token MaxSim probe on CARGO.

Idea
----
Load the *baseline* (use_afd=False) ResNet50-BoT checkpoint, extract the
layer4 spatial feature map (the GeM/pool input, 16x8 for 256x128) for every
query / gallery image, flatten it into K local "evidence" tokens, and ask:

    does a global-cosine + bidirectional MaxSim(tokens) hybrid beat plain
    global-cosine on cross-view A<->G retrieval?

If hybrid > global by +0.5~1.0 mAP, local set-matching carries real
cross-view evidence and the CVPB "Local Token MaxSim" module is worth
training. If not, drop the module (empirical kill-switch, no confound test).

Scoring (made explicit; this is the load-bearing design choice)
---------------------------------------------------------------
Ranking is done on a *combined similarity* (higher = better match):

    s_global(q,g) = cos(global_bn_q, global_bn_g)            # baseline score
    s_maxsim(q,g) = 0.5 * ( mean_k max_j cos(qtok_k, gtok_j)
                          + mean_j max_k cos(gtok_j, qtok_k) ) # bidirectional
    s_hybrid      = s_global + beta * s_maxsim

We rank by descending s_hybrid (equivalently distance = -s_hybrid).  This is
the standard MaxSim-rerank form; adding a *distance* and a *similarity*
together (as a literal reading of "global_cos + beta*maxsim") would mix signs
and is not what gives a meaningful delta, so we combine in similarity space
and rank consistently.  beta=0 reproduces the baseline global-cosine ranking,
which we use to verify the pipeline reproduces ~32.48 mAP.

All tokens are L2-normalized.  global_bn is the model's eval-mode output
(already L2-normalized BNNeck feature), so s_global here is identical to the
cosine similarity the training eval uses (eval_market ranks by 2-2*cos, i.e.
ascending cos-distance == descending cos-sim -> same order).

Run on lab-3090:
    cd /root/work/SOLIDER-REID/experiments/cargo_cvpb
    PYTHONUNBUFFERED=1 python maxsim_probe.py \
        --ckpt /root/work/SOLIDER-REID/log/cargo/afd_baseline/model_best.pth \
        2>&1 | tee /tmp/cvpb_maxsim_probe.log
"""
import os
import sys
import argparse
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# reuse the afd_reid building blocks (do NOT reimplement dataset/model/eval)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'afd_reid'))
from cargo_dataset import (CARGO, CARGOImageDataset, build_transforms,  # noqa: E402
                           filter_by_view)
from afd_model import build_model  # noqa: E402


# --------------------------------------------------------------------------- #
# feature extraction: global BN feature + layer4 spatial token map
# --------------------------------------------------------------------------- #
class TokenExtractor:
    """Hooks model.layer4 to grab its (B,2048,H,W) output during a normal
    eval forward, alongside the model's returned L2-normalized BN feature."""

    def __init__(self, model):
        self.model = model
        self._buf = {}
        # layer4 output == the map fed into GeM pool in AFDModel
        self.handle = model.layer4.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._buf['map'] = out.detach()

    def remove(self):
        self.handle.remove()

    @torch.no_grad()
    def run(self, loader, device, pool_grid=None):
        """Return (global_bn[N,2048], tokens[N,K,2048] L2-normed, pids, camids).

        pool_grid: None -> use full layer4 grid (K = H*W). Else (gh,gw) ->
        adaptive_avg_pool2d the map to gh x gw before tokenizing (K = gh*gw).
        """
        self.model.eval()
        g_list, t_list, p_list, c_list = [], [], [], []
        for batch in loader:
            imgs = batch['img'].to(device, non_blocking=True)
            # baseline: use_afd=False -> view_idx unused; eval returns normed BN feat
            gfeat = self.model(imgs)                     # (B,2048) L2-normed
            fmap = self._buf['map']                       # (B,2048,H,W)
            if pool_grid is not None:
                fmap = F.adaptive_avg_pool2d(fmap, pool_grid)
            B, C, H, W = fmap.shape
            tok = fmap.flatten(2).permute(0, 2, 1).contiguous()  # (B,K,C)
            tok = F.normalize(tok, dim=2)                 # per-token L2
            g_list.append(gfeat.cpu())
            t_list.append(tok.cpu())
            p_list.append(batch['pid'])
            c_list.append(batch['camid'])
        if not g_list:
            return (torch.empty(0), torch.empty(0),
                    np.empty(0, np.int64), np.empty(0, np.int64))
        g = torch.cat(g_list, 0)
        t = torch.cat(t_list, 0)
        p = torch.cat(p_list, 0).numpy()
        c = torch.cat(c_list, 0).numpy()
        return g, t, p, c


# --------------------------------------------------------------------------- #
# similarity matrices
# --------------------------------------------------------------------------- #
def global_sim(qg, gg):
    """Cosine similarity between L2-normed global feats -> (num_q,num_g)."""
    return (qg @ gg.t())


@torch.no_grad()
def maxsim_bidir(qt, gt, device, mem_budget_elems=120_000_000):
    """Bidirectional MaxSim similarity between token sets.

    qt:(Nq,Kq,C) gt:(Ng,Kg,C), tokens already L2-normed.
    s(q,g) = 0.5*( mean_k max_j <qtok_k,gtok_j> + mean_j max_k <gtok_j,qtok_k> )
    Returns (Nq,Ng) on cpu.

    Galleries are huge here (Ng up to ~32k) and queries tiny (~100-200), so the
    cost is dominated by Ng.  We keep ALL queries on-GPU and chunk over the
    GALLERY axis, building a (Nq,Kq,Gblk,Kg) sim block per chunk and reducing it
    immediately.  Gblk is chosen so the block has <= mem_budget_elems floats.
    """
    Nq, Kq, C = qt.shape
    Ng, Kg, _ = gt.shape
    qt_d = qt.to(device)                              # (Nq,Kq,C) small
    qt_flat = qt_d.reshape(Nq * Kq, C)                # (Nq*Kq,C)
    # block size so Nq*Kq*Gblk*Kg <= budget (and at least 1)
    per_g = max(1, Nq * Kq * Kg)
    gblk = max(1, min(Ng, mem_budget_elems // per_g))
    out = torch.empty(Nq, Ng)
    for s in range(0, Ng, gblk):
        e = min(s + gblk, Ng)
        gc = gt[s:e].to(device)                       # (g,Kg,C)
        g = gc.size(0)
        gc_flat = gc.reshape(g * Kg, C)               # (g*Kg,C)
        # (Nq*Kq, g*Kg) -> (Nq,Kq,g,Kg)
        sim = (qt_flat @ gc_flat.t()).reshape(Nq, Kq, g, Kg)
        q2g = sim.max(dim=3).values.mean(dim=1)       # (Nq,g) q-token best in g
        g2q = sim.max(dim=1).values.mean(dim=2)       # (Nq,g) g-token best in q
        out[:, s:e] = (0.5 * (q2g + g2q)).cpu()
        del sim, q2g, g2q, gc, gc_flat
    del qt_d, qt_flat
    if device == 'cuda':
        torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- #
# market-style mAP/CMC from a precomputed *distance* matrix
# (same junk removal as afd_train.eval_market; lower distance == better)
# --------------------------------------------------------------------------- #
def eval_from_distmat(distmat, q_pids, q_camids, g_pids, g_camids, max_rank=50):
    if distmat.size == 0:
        return float('nan'), float('nan'), float('nan')
    num_q, num_g = distmat.shape
    max_rank = min(max_rank, num_g)
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc, all_AP, all_INP = [], [], []
    num_valid_q = 0
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            continue
        cmc = raw_cmc.cumsum()
        pos_idx = np.where(raw_cmc == 1)[0]
        max_pos_idx = pos_idx[-1]
        all_INP.append(cmc[max_pos_idx] / (max_pos_idx + 1.0))
        cmc_clip = cmc.copy()
        cmc_clip[cmc_clip > 1] = 1
        all_cmc.append(cmc_clip[:max_rank])
        num_valid_q += 1
        num_rel = raw_cmc.sum()
        tmp = raw_cmc.cumsum()
        tmp = [x / (i + 1.0) for i, x in enumerate(tmp)]
        tmp = np.asarray(tmp) * raw_cmc
        all_AP.append(tmp.sum() / num_rel)
    if num_valid_q == 0:
        return float('nan'), float('nan'), float('nan')
    cmc = np.asarray(all_cmc).astype(np.float32).sum(0) / num_valid_q
    return float(np.mean(all_AP)) * 100, float(cmc[0]) * 100, float(np.mean(all_INP)) * 100


# --------------------------------------------------------------------------- #
# probe driver
# --------------------------------------------------------------------------- #
def build_loader(samples, args):
    tf = build_transforms(is_train=False, img_size=tuple(args.img_size))
    ds = CARGOImageDataset(samples, tf)
    return DataLoader(ds, batch_size=args.test_batch, shuffle=False,
                      num_workers=args.workers, pin_memory=True)


def grid_label(pool_grid):
    return 'full(16x8)' if pool_grid is None else f'{pool_grid[0]}x{pool_grid[1]}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',
                    default='/root/work/SOLIDER-REID/log/cargo/afd_baseline/model_best.pth')
    ap.add_argument('--data_root', default='/root/work/SOLIDER-REID/data')
    ap.add_argument('--img_size', type=int, nargs=2, default=[256, 128])
    ap.add_argument('--test_batch', type=int, default=128)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--betas', type=float, nargs='+',
                    default=[0.1, 0.3, 0.5, 1.0])
    # grids to try as token sets; 'full' == native layer4 grid (16x8=128 tokens).
    # NB: CARGO galleries are huge (g_ground=32268, g_aerial=18444); the full
    # 128-token grid needs ~34 GB just to hold gallery tokens in RAM (31 GB box),
    # so the default sweeps the meaningful pooled grids 8x4 (32 tok) and 4x2 (8 tok).
    # Pass --grids full 8x4 4x2 explicitly only on a big-RAM box.
    ap.add_argument('--grids', type=str, nargs='+',
                    default=['8x4', '4x2'],
                    help="token grids, e.g. 8x4 4x2 (or 'full' on a big-RAM box)")
    args = ap.parse_args()

    device = 'cuda'
    print('=' * 72)
    print('CVPB kill-switch #1: zero-training local-token MaxSim probe')
    print(f'  ckpt={args.ckpt}')
    print(f'  betas={args.betas}  grids={args.grids}')
    print('=' * 72)

    # ---- data ----
    dataset = CARGO(root=args.data_root, verbose=True)

    # ---- model: baseline (use_afd=False), load checkpoint ----
    model_args = types.SimpleNamespace(
        last_stride=1, pool='gem', use_afd=False,
        afd_router=False, afd_cvfc=False, afd_stage='layer1',
        router_cond_view=False, low_r=0.125, mid_r=0.30, high_drop_p=0.5)
    model = build_model(dataset.num_train_pids, model_args).to(device)
    state = torch.load(args.ckpt, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state and \
            not any(k.startswith('layer') for k in state):
        state = state['state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'  loaded checkpoint. missing={len(missing)} unexpected={len(unexpected)}')
    if missing:
        print('    [missing keys sample]', missing[:6])
    if unexpected:
        print('    [unexpected keys sample]', unexpected[:6])
    model.eval()

    extractor = TokenExtractor(model)

    # parse grids
    pool_grids = []
    for g in args.grids:
        if g.lower() == 'full':
            pool_grids.append(None)
        else:
            gh, gw = g.lower().split('x')
            pool_grids.append((int(gh), int(gw)))

    # ---- both cross-view directions ----
    q_aerial = filter_by_view(dataset.query, 'Aerial')
    q_ground = filter_by_view(dataset.query, 'Ground')
    g_aerial = filter_by_view(dataset.gallery, 'Aerial')
    g_ground = filter_by_view(dataset.gallery, 'Ground')
    directions = [('A->G', q_aerial, g_ground), ('G->A', q_ground, g_aerial)]

    # For each grid we extract tokens once per direction, then sweep betas.
    # Cache per-direction global feats so beta=0 baseline is computed once.
    # results[(grid, beta)] = {'A->G': mAP, 'G->A': mAP, ...}
    results = {}
    # also record baseline (global only) per direction
    base_map = {}

    for pg in pool_grids:
        glabel = grid_label(pg)
        print('\n' + '-' * 72)
        print(f'### token grid = {glabel} '
              f'(K = {(16 * 8) if pg is None else pg[0] * pg[1]} tokens)')
        print('-' * 72)

        per_dir = {}   # tag -> (gsim, msim, qp, qc, gp, gc)
        for tag, q, g in directions:
            ql = build_loader(q, args)
            gl = build_loader(g, args)
            qg, qt, qp, qc = extractor.run(ql, device, pool_grid=pg)
            gg, gt, gp, gc = extractor.run(gl, device, pool_grid=pg)
            gsim = global_sim(qg, gg).numpy()                      # (Nq,Ng)
            msim = maxsim_bidir(qt, gt, device).numpy()            # (Nq,Ng)
            per_dir[tag] = (gsim, msim, qp, qc, gp, gc)

            # baseline global-only (rank by -gsim == cosine distance order)
            if glabel not in base_map:
                base_map[glabel] = {}
            bmap, br1, _ = eval_from_distmat(-gsim, qp, qc, gp, gc)
            base_map[glabel][tag] = (bmap, br1)
            print(f'  [{tag}] global-only  mAP={bmap:.2f}  R1={br1:.2f}  '
                  f'(Nq={len(qp)} Ng={len(gp)})')

        # baseline mean for this grid (global feats are grid-independent ->
        # identical across grids; we print per-grid for sanity)
        bmean = np.mean([base_map[glabel][t][0] for t in ('A->G', 'G->A')])
        print(f'  [mean] global-only mAP={bmean:.2f}')

        for beta in args.betas:
            row = {}
            for tag, q, g in directions:
                gsim, msim, qp, qc, gp, gc = per_dir[tag]
                hyb = gsim + beta * msim                # combined similarity
                mAP, r1, minp = eval_from_distmat(-hyb, qp, qc, gp, gc)
                row[tag] = (mAP, r1, minp)
            mean_map = np.mean([row[t][0] for t in ('A->G', 'G->A')])
            mean_r1 = np.mean([row[t][1] for t in ('A->G', 'G->A')])
            results[(glabel, beta)] = (row, mean_map, mean_r1)
            print(f'  beta={beta:<4} | A->G mAP={row["A->G"][0]:6.2f} R1={row["A->G"][1]:6.2f}'
                  f' | G->A mAP={row["G->A"][0]:6.2f} R1={row["G->A"][1]:6.2f}'
                  f' | MEAN mAP={mean_map:6.2f} R1={mean_r1:6.2f}')

    extractor.remove()

    # ---- summary table ----
    print('\n' + '=' * 72)
    print('SUMMARY  (mean A<->G mAP; baseline global-only target ~= 32.48)')
    print('=' * 72)
    any_grid = grid_label(pool_grids[0])
    gbase_mean = np.mean([base_map[any_grid][t][0] for t in ('A->G', 'G->A')])
    print(f'{"grid":>12} | {"global":>7} | ' +
          ' | '.join(f'b={b:<4}' for b in args.betas))
    for pg in pool_grids:
        glabel = grid_label(pg)
        gb = np.mean([base_map[glabel][t][0] for t in ('A->G', 'G->A')])
        cells = []
        for b in args.betas:
            mean_map = results[(glabel, b)][1]
            cells.append(f'{mean_map:6.2f}')
        print(f'{glabel:>12} | {gb:7.2f} | ' + ' | '.join(cells))

    # ---- verdict ----
    best_key = max(results, key=lambda k: results[k][1])
    best_mean = results[best_key][1]
    delta = best_mean - gbase_mean
    print('-' * 72)
    print(f'baseline global-only mean mAP = {gbase_mean:.2f}')
    print(f'best hybrid = grid {best_key[0]} beta {best_key[1]} -> '
          f'mean mAP {best_mean:.2f}  (delta {delta:+.2f})')
    if delta >= 0.5:
        print(f'VERDICT: PASS  (+{delta:.2f} >= +0.5) -> local MaxSim carries '
              f'cross-view evidence; train the Local Token MaxSim module.')
    else:
        print(f'VERDICT: FAIL  ({delta:+.2f} < +0.5) -> local MaxSim adds little '
              f'on the trained baseline; drop/replace the module.')
    print('=' * 72)


if __name__ == '__main__':
    main()
