#!/usr/bin/env python3
"""Ambiguous Query-Bag ReID — ZERO-TRAINING kill-switch (frozen feats + numpy).

Re-frame under test (litreview2/explore20/clean/d_20.txt, opportunity 1):
    Standard ReID assumes the query is a single correct target image; multi-query
    assumes every query image is the same correct ID. Real deployment instead gives
    a *bag* of candidate crops (tracking drift / wrong boxes / neighbouring person /
    operator mistakes) whose purity is unknown. People assume "more query support =
    more stable"; the hidden variable is query-bag PURITY: feeding wrong images can
    be worse than a single clean image. Re-define single-query ReID as
    WEAK-TARGET-EVIDENCE BAG retrieval.

Mechanism under test (zero-training): Target-Consensus Query Aggregation.
    Each bag image independently retrieves its top-L gallery. Build a bag x bag
    agreement graph (edge = top-L overlap, i.e. how much two bag images agree on
    *who* the gallery target is). Take the largest / densest agreement subset (the
    consensus set) and fuse ONLY those images (trimmed mean). The intuition: true
    targets agree on the same gallery identity; contaminants point elsewhere and fall
    out of the consensus component.

GO  (must clear ALL):
    * at 25-50% hard contamination, plain AVERAGE multi-query loses > 10 mAP vs 0%;
    * Target-Consensus recovers >= half of that average drop;
    * AND Target-Consensus beats k-reciprocal / camera / single-best by >= 2 mAP.
NO-GO:
    * single-best OR k-reciprocal already neutralises the contamination (=> no new
      method needed, the trivial baseline solved it -- the HUBNESS §7.6 lesson:
      do NOT let a trivial baseline win and still call it success);
    * OR contamination causes no clear protocol gap.

NOTHING is trained: frozen cached features + numpy only. Two datasets (market + od).

Run on lab-3090-d (features already cached by the hubness script):
    cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 \
      /root/miniconda3/envs/solider-reid/bin/python \
      experiments/cargo_cvpb/cvpb_querybag_killswitch.py \
      --cache_feat /tmp/hub_market_feats.npz --dataset market1501 \
      2>&1 | tee /tmp/cvpb_querybag_market.log
    # od: --cache_feat /tmp/hub_oduke_feats.npz --dataset occluded_duke
"""
import os, sys, time, argparse, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--cache_feat', default='/tmp/hub_market_feats.npz',
                help='cached frozen features (q_feat/q_pid/q_cam, g_feat/g_pid/g_cam)')
ap.add_argument('--dataset', default='market1501', help='label only (headers)')
ap.add_argument('--bag_m', type=int, default=4,
                help='bag size at 0%% contamination = number of TRUE-target images')
ap.add_argument('--contam_rates', type=float, nargs='+', default=[0.0, 0.25, 0.50, 0.75],
                help='c/(m+c) contamination fractions to sweep')
ap.add_argument('--topL', type=int, default=50, help='consensus per-image retrieval depth L')
ap.add_argument('--consensus_overlap_thr', type=float, default=0.10,
                help='min top-L Jaccard overlap to draw an agreement edge')
ap.add_argument('--cons_thr_grid', type=float, nargs='+',
                default=[0.05, 0.10, 0.20, 0.30, 0.50],
                help='Target-Consensus overlap-threshold sweep (best kept = oracle upper bound)')
ap.add_argument('--cons_topL_grid', type=int, nargs='+', default=[20, 50, 100],
                help='Target-Consensus top-L retrieval-depth sweep')
ap.add_argument('--trim_frac', type=float, default=0.25,
                help='trimmed-mean: fraction trimmed each tail (image-level / dim-level)')
ap.add_argument('--cam_gamma', type=float, nargs='+',
                default=[0.05, 0.1, 0.2, 0.3, 0.5], help='camera-aware down-weight sweep')
ap.add_argument('--n_bags_per_id', type=int, default=1,
                help='how many bags to sample per query identity')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--smoke', type=int, default=0, help='cap #query-identities for a fast run')
cli = ap.parse_args()
RNG = np.random.RandomState(cli.seed)


# =========================================================================== #
# EVAL: standard market/od protocol — drop gallery with (same pid AND same cam)
# as the query ANCHOR. ground-truth pid/cam = anchor's. distmat row per bag.
# =========================================================================== #
def eval_map(distmat, q_pid, q_cam, g_pid, g_cam, max_rank=20):
    num_q = distmat.shape[0]
    indices = np.argsort(distmat, axis=1)
    all_AP, all_cmc = [], []
    for i in range(num_q):
        order = indices[i]
        remove = (g_pid[order] == q_pid[i]) & (g_cam[order] == q_cam[i])
        keep = ~remove
        m = (g_pid[order][keep] == q_pid[i]).astype(np.int32)
        if not m.any():
            continue
        cmc = m.cumsum(); cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        tmp = m.cumsum()
        prec = tmp / (np.arange(len(m)) + 1.0)
        all_AP.append((prec * m).sum() / m.sum())
    if not all_AP:
        return dict(mAP=float('nan'), r1=float('nan'), r5=float('nan'), r10=float('nan'), nq=0)
    all_cmc = np.asarray(all_cmc).mean(0)
    return dict(mAP=float(np.mean(all_AP)) * 100, r1=float(all_cmc[0]) * 100,
                r5=float(all_cmc[4]) * 100, r10=float(all_cmc[9]) * 100, nq=len(all_AP))


# --------------------------------------------------------------------------- #
# k-reciprocal re-ranking (Zhong 2017), numpy. Operates on a set of "query"
# feature vectors (here: the fused bag-average vectors) vs gallery. Returns a
# (Nq,Ng) re-ranked distance matrix. (camera_aware: drop same-cam from recip set)
# --------------------------------------------------------------------------- #
def kreciprocal_rerank(qf, gf, q_cam, g_cam, k1=20, k2=6, lam=0.3, camera_aware=False):
    qf = qf.astype(np.float32); gf = gf.astype(np.float32)
    Nq, Ng = qf.shape[0], gf.shape[0]
    allf = np.concatenate([qf, gf], 0)
    cams = np.concatenate([q_cam, g_cam], 0)
    orig = 2.0 - 2.0 * (allf @ allf.T)          # squared-euclid on unit sphere
    orig = np.maximum(orig, 0.0)
    N = Nq + Ng
    initial_rank = np.argsort(orig, axis=1).astype(np.int32)
    V = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        fwd = initial_rank[i, :k1 + 1]
        recip = [c for c in fwd if i in initial_rank[c, :k1 + 1]]
        recip = np.array(recip, dtype=np.int32) if len(recip) else np.array([i], dtype=np.int32)
        recip_exp = list(recip)
        for cand in recip:
            cand_half = initial_rank[cand, :int(np.around(k1 / 2.0)) + 1]
            if len(np.intersect1d(cand_half, recip)) > 2.0 / 3.0 * len(cand_half):
                recip_exp.extend(cand_half.tolist())
        recip_exp = np.unique(np.array(recip_exp, dtype=np.int32))
        if camera_aware:
            recip_exp = recip_exp[(cams[recip_exp] != cams[i]) | (recip_exp == i)]
            if len(recip_exp) == 0:
                recip_exp = np.array([i], dtype=np.int32)
        w = np.exp(-orig[i, recip_exp])
        V[i, recip_exp] = (w / w.sum()).astype(np.float32)
    if k2 > 1:
        V_qe = np.zeros_like(V)
        for i in range(N):
            V_qe[i] = V[initial_rank[i, :k2]].mean(0)
        V = V_qe
    invIndex = [np.where(V[:, j] != 0)[0] for j in range(N)]
    jaccard = np.zeros((Nq, Ng), dtype=np.float32)
    for i in range(Nq):
        nz = np.where(V[i] != 0)[0]
        minsum = np.zeros(N, dtype=np.float32)
        for j in nz:
            cols = invIndex[j]
            minsum[cols] += np.minimum(V[i, j], V[cols, j])
        jd = 1.0 - minsum / (2.0 - minsum + 1e-12)
        jaccard[i] = jd[Nq:]
    final = (1.0 - lam) * jaccard + lam * orig[:Nq, Nq:]
    return final


# =========================================================================== #
# BAG CONSTRUCTION
# =========================================================================== #
def build_bags(qf, q_pid, q_cam, gf, g_pid, g_cam, c_count, m_true):
    """For each query identity, build ONE bag (or n_bags_per_id) with:
        - 1 ANCHOR true-target query image (defines pid/cam for the junk rule + GT)
        - (m_true - 1) extra TRUE-target images (same pid; other query images,
          else extra gallery images of the same pid w/ a different camera than anchor;
          else resample the anchor pool)
        - c_count HARD CONTAMINANTS = gallery features whose pid != anchor pid,
          drawn from the anchor's baseline cosine top-20 (already rank high -> they
          actively pull the fused feature toward a wrong identity).

    Returns a list of dicts:
        bag_feat (B,D) unit-normed | anchor_pid | anchor_cam | n_true | n_contam
        is_true (B,) bool          | (B = m_true + c_count)
    Bag features are L2-normed individual image vectors (NO fusion yet)."""
    qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
    gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)

    # group query indices by pid
    pid2q = {}
    for i, p in enumerate(q_pid):
        pid2q.setdefault(int(p), []).append(i)
    pid2g = {}
    for j, p in enumerate(g_pid):
        pid2g.setdefault(int(p), []).append(j)

    uniq_pids = sorted(pid2q.keys())
    if cli.smoke > 0:
        uniq_pids = uniq_pids[:cli.smoke]

    bags = []
    for p in uniq_pids:
        qidx = pid2q[p]
        for _b in range(cli.n_bags_per_id):
            anchor = qidx[RNG.randint(len(qidx))]
            a_cam = int(q_cam[anchor])
            # ---- true-target images (anchor + extras) ----
            true_feats = [qf[anchor]]
            n_extra = m_true - 1
            # pool of other true images: other query images of same pid (excl anchor)
            #  + same-pid gallery images with a DIFFERENT camera than the anchor
            other_q = [i for i in qidx if i != anchor]
            same_pid_g = pid2g.get(p, [])
            cross_cam_g = [j for j in same_pid_g if int(g_cam[j]) != a_cam]
            pool_feats = [qf[i] for i in other_q] + [gf[j] for j in cross_cam_g]
            if n_extra > 0:
                if len(pool_feats) >= n_extra:
                    sel = RNG.choice(len(pool_feats), size=n_extra, replace=False)
                else:
                    sel = RNG.choice(len(pool_feats), size=n_extra, replace=True) \
                        if len(pool_feats) > 0 else []
                for s in sel:
                    true_feats.append(pool_feats[s])
                # if pool empty (rare), pad by repeating anchor
                while len(true_feats) < m_true:
                    true_feats.append(qf[anchor])
            true_feats = np.stack(true_feats, 0)

            # ---- hard contaminants from anchor's baseline top-20 wrong-ID gallery ----
            contam_feats = []
            if c_count > 0:
                sims = gf @ qf[anchor]
                order = np.argsort(-sims)
                # walk down the ranking, take WRONG-ID gallery (different pid),
                # also skip junk same-cam-same-pid (n/a since pid differs).
                cand = []
                for j in order:
                    if int(g_pid[j]) != p:
                        cand.append(j)
                    if len(cand) >= 20:
                        break
                if len(cand) == 0:                       # degenerate; skip bag
                    continue
                if len(cand) >= c_count:
                    pick = RNG.choice(len(cand), size=c_count, replace=False)
                else:
                    pick = RNG.choice(len(cand), size=c_count, replace=True)
                for s in pick:
                    contam_feats.append(gf[cand[s]])
                contam_feats = np.stack(contam_feats, 0)

            if len(contam_feats):
                bag_feat = np.concatenate([true_feats, contam_feats], 0)
                is_true = np.array([True] * len(true_feats) + [False] * len(contam_feats))
            else:
                bag_feat = true_feats
                is_true = np.array([True] * len(true_feats))
            bag_feat = bag_feat / (np.linalg.norm(bag_feat, axis=1, keepdims=True) + 1e-12)
            bags.append(dict(bag_feat=bag_feat.astype(np.float32), pid=p, cam=a_cam,
                             n_true=len(true_feats), n_contam=len(contam_feats),
                             is_true=is_true))
    return bags


# =========================================================================== #
# FUSION STRATEGIES   bag(B,D) -> single query vector (D,)  OR  score(Ng,)
# Convention: produce a (D,) FUSED vector when possible; for single-best we
# produce a (Ng,) similarity directly (max over bag). We standardise everything
# to a per-bag DISTANCE row over gallery, then stack into a (Nbag, Ng) distmat.
# =========================================================================== #
def fuse_average(bag, gf):
    v = bag.mean(0)
    v /= (np.linalg.norm(v) + 1e-12)
    return 1.0 - gf @ v                                  # distance row

def fuse_median(bag, gf):
    v = np.median(bag, axis=0)
    v /= (np.linalg.norm(v) + 1e-12)
    return 1.0 - gf @ v

def fuse_trimmed(bag, gf, trim=0.25):
    """Image-level trimmed mean: drop the bag images farthest from the bag medoid,
    trimming `trim` fraction (>=1 each side only if enough images)."""
    B = bag.shape[0]
    centroid = bag.mean(0); centroid /= (np.linalg.norm(centroid) + 1e-12)
    d2c = 1.0 - bag @ centroid
    ntrim = int(np.floor(trim * B))
    keep = np.argsort(d2c)[:B - ntrim] if ntrim > 0 and B - ntrim >= 1 else np.arange(B)
    v = bag[keep].mean(0); v /= (np.linalg.norm(v) + 1e-12)
    return 1.0 - gf @ v

def fuse_single_best(bag, gf):
    """Multi-query single-best (a.k.a. min-distance): score(g) = max_i cos(bag_i, g).
    This is the standard 'pick the closest bag image per gallery' baseline that is
    naturally robust to contaminants that are simply far from every gallery target."""
    sims = bag @ gf.T                                    # (B, Ng)
    return 1.0 - sims.max(0)                             # distance row

def bag_agreement_matrix(bag, gf, topL):
    """A[i,j] = top-L Jaccard overlap between bag image i and j (how much they agree
    on WHICH gallery items are the target). Diagonal = 0."""
    B = bag.shape[0]
    sims = bag @ gf.T                                    # (B, Ng)
    topsets = [set(np.argpartition(-sims[i], topL)[:topL].tolist()) for i in range(B)]
    A = np.zeros((B, B))
    for i in range(B):
        for j in range(i + 1, B):
            inter = len(topsets[i] & topsets[j])
            union = len(topsets[i] | topsets[j])
            A[i, j] = A[j, i] = (inter / union if union else 0.0)
    return A

def consensus_select(bag, gf, topL, overlap_thr, mode='medoid'):
    """Target-Consensus selection. Two modes:
      'component': largest connected component of the (A>=thr) agreement graph
                   (ties -> densest). [the literal design draft]
      'medoid'   : seed = bag image with highest summed agreement (most 'agreed-with'
                   = most likely a true target since true targets co-retrieve the same
                   gallery id); grow the consensus by adding every image whose mean
                   agreement to the current set >= thr. This is more robust when hard
                   contaminants weakly inter-agree and would bridge a component.
    Returns consensus indices (>=1)."""
    B = bag.shape[0]
    if B <= 1:
        return np.arange(B, dtype=int)
    A = bag_agreement_matrix(bag, gf, topL)
    if mode == 'component':
        adj = A >= overlap_thr
        seen = np.zeros(B, bool); comps = []
        for s in range(B):
            if seen[s]:
                continue
            stack = [s]; comp = []; seen[s] = True
            while stack:
                u = stack.pop(); comp.append(u)
                for v in range(B):
                    if adj[u, v] and not seen[v]:
                        seen[v] = True; stack.append(v)
            comps.append(comp)
        def dens(c):
            if len(c) < 2:
                return 0.0
            sub = A[np.ix_(c, c)]; iu = np.triu_indices(len(c), 1)
            return float(sub[iu].mean())
        comps.sort(key=lambda c: (len(c), dens(c)), reverse=True)
        return np.array(comps[0], dtype=int)
    # medoid mode
    seed = int(np.argmax(A.sum(1)))
    consensus = [seed]
    remaining = [i for i in range(B) if i != seed]
    changed = True
    while changed and remaining:
        changed = False
        for i in list(remaining):
            mean_agree = A[i, consensus].mean()
            if mean_agree >= overlap_thr:
                consensus.append(i); remaining.remove(i); changed = True
    return np.array(consensus, dtype=int)

def consensus_select_fast(top_idx, overlap_thr, mode='medoid'):
    """Same selection as consensus_select but takes a precomputed (B, L) array of
    each bag image's top-L gallery indices (sorted desc). Builds the B x B Jaccard
    agreement matrix from those sets. Avoids recomputing bag @ gf.T per sweep config."""
    B = top_idx.shape[0]
    if B <= 1:
        return np.arange(B, dtype=int)
    topsets = [set(top_idx[i].tolist()) for i in range(B)]
    A = np.zeros((B, B))
    for i in range(B):
        for j in range(i + 1, B):
            inter = len(topsets[i] & topsets[j]); union = len(topsets[i] | topsets[j])
            A[i, j] = A[j, i] = (inter / union if union else 0.0)
    if mode == 'component':
        adj = A >= overlap_thr
        seen = np.zeros(B, bool); comps = []
        for s in range(B):
            if seen[s]:
                continue
            stack = [s]; comp = []; seen[s] = True
            while stack:
                u = stack.pop(); comp.append(u)
                for v in range(B):
                    if adj[u, v] and not seen[v]:
                        seen[v] = True; stack.append(v)
            comps.append(comp)
        def dens(c):
            if len(c) < 2:
                return 0.0
            sub = A[np.ix_(c, c)]; iu = np.triu_indices(len(c), 1)
            return float(sub[iu].mean())
        comps.sort(key=lambda c: (len(c), dens(c)), reverse=True)
        return np.array(comps[0], dtype=int)
    seed = int(np.argmax(A.sum(1)))
    consensus = [seed]; remaining = [i for i in range(B) if i != seed]
    changed = True
    while changed and remaining:
        changed = False
        for i in list(remaining):
            if A[i, consensus].mean() >= overlap_thr:
                consensus.append(i); remaining.remove(i); changed = True
    return np.array(consensus, dtype=int)


def fuse_consensus(bag, gf, topL, overlap_thr, trim=0.25, mode='medoid'):
    keep = consensus_select(bag, gf, topL, overlap_thr, mode=mode)
    sub = bag[keep]
    if sub.shape[0] >= 3:                                # trimmed mean within consensus
        return fuse_trimmed(sub, gf, trim=trim)
    v = sub.mean(0); v /= (np.linalg.norm(v) + 1e-12)
    return 1.0 - gf @ v


# =========================================================================== #
# RUN one (dataset, contamination-rate) cell across all strategies
# =========================================================================== #
def run_cell(bags, gf, g_pid, g_cam):
    """Given a list of bags (all at one contamination rate), compute distmats for
    each strategy and eval. Returns dict strategy -> metrics."""
    Nb = len(bags)
    Ng = gf.shape[0]
    bag_pid = np.array([b['pid'] for b in bags])
    bag_cam = np.array([b['cam'] for b in bags])

    # --- simple per-bag fusions producing distance rows ---
    D = {s: np.empty((Nb, Ng), np.float32) for s in
         ['avg', 'single-best', 'median', 'trimmed']}
    fused_avg_vecs = np.empty((Nb, gf.shape[1]), np.float32)   # for k-recip / camera
    for bi, b in enumerate(bags):
        bag = b['bag_feat']
        D['avg'][bi] = fuse_average(bag, gf)
        D['single-best'][bi] = fuse_single_best(bag, gf)
        D['median'][bi] = fuse_median(bag, gf)
        D['trimmed'][bi] = fuse_trimmed(bag, gf, trim=cli.trim_frac)
        va = bag.mean(0); va /= (np.linalg.norm(va) + 1e-12)
        fused_avg_vecs[bi] = va

    out = {}
    for s in ['avg', 'single-best', 'median', 'trimmed']:
        out[s] = eval_map(D[s], bag_pid, bag_cam, g_pid, g_cam)

    # --- Target-Consensus: SWEEP (mode, overlap_thr, topL); keep the BEST config so
    #     the proposed method gets its strongest possible shot (oracle-tuned upper
    #     bound -- if it cannot clear the bar even here, it is dead). Also track the
    #     selection quality (mean consensus size & purity) for the best config. ---
    # Precompute per-bag top-Lmax sims ONCE (agreement matrix only depends on topL).
    Lmax = max(cli.cons_topL_grid)
    bag_top_idx = []   # per bag: (B, Lmax) gallery indices sorted by sim desc
    for b in bags:
        sims = b['bag_feat'] @ gf.T                      # (B, Ng)
        part = np.argpartition(-sims, Lmax - 1, axis=1)[:, :Lmax]
        rows = np.arange(sims.shape[0])[:, None]
        order = np.argsort(-sims[rows, part], axis=1)
        bag_top_idx.append(part[rows, order])            # (B, Lmax) desc
    cons_grid = []
    for mode in ['medoid', 'component']:
        for thr in cli.cons_thr_grid:
            for L in cli.cons_topL_grid:
                Dc = np.empty((Nb, Ng), np.float32)
                sizes, purity = [], []
                for bi, b in enumerate(bags):
                    bag = b['bag_feat']
                    keep = consensus_select_fast(bag_top_idx[bi][:, :L], thr, mode=mode)
                    sizes.append(len(keep)); purity.append(float(b['is_true'][keep].mean()))
                    sub = bag[keep]
                    if sub.shape[0] >= 3:
                        Dc[bi] = fuse_trimmed(sub, gf, trim=cli.trim_frac)
                    else:
                        v = sub.mean(0); v /= (np.linalg.norm(v) + 1e-12)
                        Dc[bi] = 1.0 - gf @ v
                r = eval_map(Dc, bag_pid, bag_cam, g_pid, g_cam)
                cons_grid.append(dict(mode=mode, thr=thr, L=L, metrics=r,
                                      size=float(np.mean(sizes)),
                                      purity=float(np.mean(purity))))
    best_cons = max(cons_grid, key=lambda d: (d['metrics']['mAP']
                                              if not np.isnan(d['metrics']['mAP']) else -1))
    out['consensus'] = best_cons['metrics']

    # --- k-reciprocal on the bag-average fused vectors (strong test-time baseline) ---
    rr = kreciprocal_rerank(fused_avg_vecs, gf, bag_cam, g_cam, k1=20, k2=6, lam=0.3,
                            camera_aware=False)
    out['k-recip'] = eval_map(rr, bag_pid, bag_cam, g_pid, g_cam)

    # --- camera-aware baseline on bag-average: down-weight same-cam-as-anchor gallery ---
    sim_avg = fused_avg_vecs @ gf.T
    same_cam = (g_cam[None, :] == bag_cam[:, None]).astype(np.float32)
    cam_best = dict(g=0.0, **out['avg'])
    for gamma in cli.cam_gamma:
        r = eval_map(1.0 - (sim_avg - gamma * same_cam), bag_pid, bag_cam, g_pid, g_cam)
        if not np.isnan(r['mAP']) and r['mAP'] > cam_best['mAP']:
            cam_best = dict(g=gamma, **r)
    out['camera'] = cam_best

    diag = dict(consensus_best_cfg=f"mode={best_cons['mode']} thr={best_cons['thr']} L={best_cons['L']}",
                consensus_mean_size=best_cons['size'],
                consensus_mean_purity=best_cons['purity'],
                n_bags=Nb)
    return out, diag


def main():
    DS = cli.dataset
    print("#" * 84)
    print(f"# AMBIGUOUS QUERY-BAG KILL-SWITCH  dataset={DS}  cache={cli.cache_feat}")
    print(f"#   bag_m(true)={cli.bag_m}  contam_rates={cli.contam_rates}  topL={cli.topL}  "
          f"overlap_thr={cli.consensus_overlap_thr}  trim={cli.trim_frac}  seed={cli.seed}")
    print("#" * 84)

    z = np.load(cli.cache_feat, allow_pickle=True)
    qf, q_pid, q_cam = z['q_feat'].astype(np.float32), z['q_pid'], z['q_cam']
    gf, g_pid, g_cam = z['g_feat'].astype(np.float32), z['g_pid'], z['g_cam']
    keep_g = g_pid != -1
    gf, g_pid, g_cam = gf[keep_g], g_pid[keep_g], g_cam[keep_g]
    gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
    print(f"[data] Nq={qf.shape[0]} Ng={gf.shape[0]} dim={qf.shape[1]} "
          f"#q-pids={len(np.unique(q_pid))} #g-pids={len(np.unique(g_pid))}")

    # sanity: single-image standard mAP (anchor only) for reference
    t0 = time.time()
    STRATS = ['avg', 'single-best', 'median', 'trimmed', 'consensus', 'k-recip', 'camera']
    results = {}   # contam -> strat -> metrics
    diags = {}
    for cr in cli.contam_rates:
        # split bag_m TRUE + c CONTAM so that c/(m+c) ~= cr with FIXED true count = bag_m
        m_true = cli.bag_m
        if cr <= 0:
            c_count = 0
        else:
            c_count = int(round(cr * m_true / (1.0 - cr)))
        actual = c_count / (m_true + c_count) if (m_true + c_count) else 0.0
        bags = build_bags(qf, q_pid, q_cam, gf, g_pid, g_cam, c_count, m_true)
        res, diag = run_cell(bags, gf, g_pid, g_cam)
        results[cr] = res; diags[cr] = diag
        print(f"\n=== contamination target={cr:.0%}  (m_true={m_true} c={c_count} "
              f"actual={actual:.1%})  bags={diag['n_bags']}  "
              f"[best-consensus {diag['consensus_best_cfg']} size={diag['consensus_mean_size']:.2f} "
              f"purity={diag['consensus_mean_purity']:.2f}]  "
              f"({time.time()-t0:.0f}s) ===")
        print(f"  {'strategy':<13} {'mAP':>7} {'R1':>7} {'R5':>7} {'R10':>7}  nq")
        for s in STRATS:
            r = res[s]
            tag = ''
            if s == 'camera':
                tag = f"  (gamma={res['camera'].get('g',0.0)})"
            print(f"  {s:<13} {r['mAP']:7.2f} {r['r1']:7.2f} {r['r5']:7.2f} {r['r10']:7.2f}  "
                  f"{r['nq']}{tag}")

    # ======================================================================= #
    # SUMMARY TABLES (raw numbers) + VERDICT
    # ======================================================================= #
    print("\n" + "#" * 84)
    print(f"# SUMMARY — {DS}  (rows = contamination rate, cols = strategy)")
    print("#" * 84)
    for metric in ['mAP', 'r1']:
        print(f"\n[{metric}]  bag true-count m={cli.bag_m}")
        hdr = f"  {'contam':<8}" + "".join(f"{s:>13}" for s in STRATS)
        print(hdr)
        for cr in cli.contam_rates:
            row = f"  {cr:<8.0%}" + "".join(f"{results[cr][s][metric]:13.2f}" for s in STRATS)
            print(row)

    # ---- verdict computation ----
    print("\n" + "#" * 84)
    print(f"# VERDICT — {DS}")
    print("#" * 84)
    clean = results[0.0]['avg']['mAP']
    verdict_lines = []
    go_flags = []
    for cr in cli.contam_rates:
        if cr <= 0:
            continue
        avg = results[cr]['avg']['mAP']
        avg_drop = clean - avg
        cons = results[cr]['consensus']['mAP']
        cons_recovery = cons - avg                       # how much consensus claws back
        half_drop = 0.5 * avg_drop
        sb = results[cr]['single-best']['mAP']
        kr = results[cr]['k-recip']['mAP']
        cam = results[cr]['camera']['mAP']
        cons_vs_sb = cons - sb
        cons_vs_kr = cons - kr
        cons_vs_cam = cons - cam
        big_drop = avg_drop > 10.0
        recovers_half = cons_recovery >= half_drop and avg_drop > 0
        beats_baselines = (cons_vs_sb >= 2.0) and (cons_vs_kr >= 2.0) and (cons_vs_cam >= 2.0)
        trivial_solved = (sb >= avg + 0.5 * avg_drop) or (kr >= avg + 0.5 * avg_drop)
        cell_go = big_drop and recovers_half and beats_baselines and not trivial_solved
        go_flags.append((cr, cell_go))
        print(f"\n  contamination {cr:.0%}:")
        print(f"    avg mAP {avg:.2f}  (drop from clean {clean:.2f} = -{avg_drop:.2f})  "
              f"{'>10 OK' if big_drop else '<=10  (no big protocol gap)'}")
        print(f"    consensus {cons:.2f}  recovery vs avg = +{cons_recovery:.2f}  "
              f"(half-drop bar = {half_drop:.2f}) {'OK' if recovers_half else 'FAIL'}")
        print(f"    single-best {sb:.2f} (cons-sb {cons_vs_sb:+.2f})  "
              f"k-recip {kr:.2f} (cons-kr {cons_vs_kr:+.2f})  "
              f"camera {cam:.2f} (cons-cam {cons_vs_cam:+.2f})")
        print(f"    beats sb&kr&cam by>=2: {'OK' if beats_baselines else 'FAIL'}   "
              f"trivial-baseline-already-solved: {'YES (NO-GO)' if trivial_solved else 'no'}")
        print(f"    => cell {'GO' if cell_go else 'NO-GO'}")

    any_go = any(f for _, f in go_flags)
    print(f"\n  >>> {DS} OVERALL: "
          f"{'GO (some cell clears all bars)' if any_go else 'NO-GO'}")
    print(f"      (GO requires: avg drop>10 AND consensus recovers>=half AND "
          f"consensus beats single-best/k-recip/camera by>=2, none trivially solved)")
    print(f"\n[done] {DS} query-bag kill-switch complete ({time.time()-t0:.0f}s).")
    # machine-readable dump
    dump = {cr: {s: results[cr][s] for s in STRATS} for cr in cli.contam_rates}
    print("JSON_DUMP " + json.dumps({'dataset': DS, 'clean_avg_mAP': clean,
                                     'results': dump, 'diags': diags}))


if __name__ == '__main__':
    main()
