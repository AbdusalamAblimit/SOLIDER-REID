#!/usr/bin/env python3
"""Gallery Hubness ReID — ZERO-TRAINING kill-switch.

Re-frame under test (HUBNESS_KILLSWITCH_DESIGN.md):
    Strong ReID failure is NOT a pairwise-similarity disease ("this query did not
    match well") but a many-to-one GRAPH-TOPOLOGY disease: a FEW gallery samples
    become attraction points (hubs) for MANY different-identity queries. ReID is a
    DIRECTED kNN-graph retrieval; the hidden variable is the gallery's NEGATIVE
    in-degree / hub mass (NOT hard-negative distance: a hard negative is close to ONE
    anchor; a hub is close to MANY different identities -> a global mis-attraction
    point).

Core quantities (k in {5,10,20}):
    H_k(g) = #{ q | g in top-k(q) AND y_g != y_q }      gallery g NEGATIVE in-degree
    M(q)   = sum_{g in topk(q), y_g != y_q} H_k(g)       query-level hub mass

Core tests:
    1. hub concentration: do the top-1% highest-H_k gallery items absorb >=20-30% of
       all false top-1 / false top-10 hits? (uniform expectation = 1%)
    2. M(q) explains per-query AP error, vs feature-norm / top1-margin / camera-pair /
       #gallery-positives (partial correlation must survive).
    3. zero-training intervention score'(q,g) = cosine - lambda*log(1+H_k(g)); sweep
       lambda, mAP/R1 must rise AND the gain must concentrate on high-M(q) queries.
    4. two datasets: Market (+ MSMT if a strong ckpt exists).

Destructive controls (decide life/death):
    D1 shuffle H_k -> intervention gain must vanish.
    D2 camera-correction equivalence: replace the intervention with a pure
       same-camera down-weight (and a CA-Jaccard-style k-reciprocal camera-aware
       re-rank). If camera correction lifts mAP just as much, the hub has no
       independent value (collides with DART3 / CA-Jaccard).
    D3 control norm + top1-margin + camera: partial correlation of M(q) must stay
       significant.
    D4 NEGATIVE in-degree vs ALL in-degree (incl. same-ID). The key signal MUST be
       the negative (cross-ID mis-attraction), not "popular sample" total in-degree.

NOTHING is trained: frozen ckpt + torch.no_grad + numpy.

Run on lab-3090-d (Market, strongest ckpt exp260b mAP 94.4):
    cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 python3 \
      experiments/cargo_cvpb/cvpb_hubness_killswitch.py \
      --config configs/market/pose_psg_lgpa_gcn_base.yml \
      --ckpt   log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth \
      --dataset market1501 \
      --cache_feat /tmp/hub_market_feats.npz 2>&1 | tee /tmp/cvpb_hubness_market.log
    # smoke first: add  --smoke 200
"""
import os, sys, time, argparse, json
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.abspath(os.path.join(_here, '..', '..'))   # repo root = .../SOLIDER-REID
sys.path.insert(0, _repo)

ap = argparse.ArgumentParser()
ap.add_argument('--config', default='configs/market/pose_psg_lgpa_gcn_base.yml')
ap.add_argument('--ckpt', default='log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth')
ap.add_argument('--dataset', default='market1501', help='label only (for headers/cache name)')
ap.add_argument('--cache_feat', default='/tmp/hub_market_feats.npz',
                help='dump/reuse extracted frozen global features (skip extraction if reused)')
ap.add_argument('--reuse_feat', action='store_true', help='reuse --cache_feat if present')
ap.add_argument('--smoke', type=int, default=0, help='if >0 cap #query for a fast smoke run')
ap.add_argument('--ks', type=int, nargs='+', default=[5, 10, 20], help='top-k for H_k')
ap.add_argument('--k_main', type=int, default=10, help='which k drives M(q) and intervention')
ap.add_argument('--lambdas', type=float, nargs='+',
                default=[0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
ap.add_argument('--seed', type=int, default=42)
cli = ap.parse_args()
np.random.seed(cli.seed)
RNG = np.random.RandomState(cli.seed)


# =========================================================================== #
# 1. FEATURE EXTRACTION  (frozen ckpt, POSE_TEST_FEAT='global', single vector)
# =========================================================================== #
def extract_features():
    import torch
    import torch.nn.functional as F
    from config import cfg
    from datasets import make_dataloader
    from model import make_model

    cfg.merge_from_file(os.path.join(_repo, cli.config))
    # force clean single-vector global feature; frozen; standard cosine eval.
    # NECK_FEAT='after' -> BN-neck feature (the trained eval feature); PSG still gates
    # the backbone so this is the REAL pose-aware global vector.
    cfg.merge_from_list([
        'TEST.WEIGHT', os.path.join(_repo, cli.ckpt),
        'MODEL.POSE_TEST_FEAT', 'global',
        'TEST.NECK_FEAT', 'after',
        'TEST.FEAT_NORM', 'yes',
        'TEST.IMS_PER_BATCH', 64,
    ])
    cfg.freeze()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = \
        make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num,
                       view_num=view_num, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
    model.load_param(cfg.TEST.WEIGHT)
    model = model.cuda().eval()
    print(f"[extract] loaded {cli.ckpt}; POSE_TEST_FEAT=global; num_query={num_query}", flush=True)

    feats, pids, camids, names = [], [], [], []
    t0 = time.time()
    use_pose = cfg.MODEL.POSE_ENABLED
    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            imgs = batch[0].cuda(non_blocking=True)
            b_pids = batch[1]
            b_camids_t = batch[3]
            b_views = batch[4]
            img_paths = batch[5]
            pose_dict = batch[6] if (use_pose and len(batch) > 6) else None
            if pose_dict is not None:
                pose_dict = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                             for k, v in pose_dict.items()}
                out = model(imgs, cam_label=b_camids_t.cuda(), view_label=b_views.cuda(),
                            pose_dict=pose_dict)
            else:
                out = model(imgs, cam_label=b_camids_t.cuda(), view_label=b_views.cuda())
            feat = out[0] if isinstance(out, (tuple, list)) else out
            assert torch.is_tensor(feat) and feat.dim() == 2, \
                f"expected single global vector, got {type(feat)} {getattr(feat,'shape',None)}"
            feat = F.normalize(feat, p=2, dim=1)
            feats.append(feat.cpu().numpy().astype(np.float32))
            pids.extend([int(x) for x in b_pids])
            camids.extend([int(x) for x in (b_camids_t.tolist())])
            names.extend([os.path.basename(p) for p in img_paths])
            if bi % 20 == 0:
                print(f"  [extract] batch {bi}/{len(val_loader)} ({time.time()-t0:.0f}s)", flush=True)

    feats = np.concatenate(feats, 0)
    pids = np.asarray(pids); camids = np.asarray(camids); names = np.asarray(names)
    q = dict(feat=feats[:num_query], pid=pids[:num_query], cam=camids[:num_query],
             name=names[:num_query])
    g = dict(feat=feats[num_query:], pid=pids[num_query:], cam=camids[num_query:],
             name=names[num_query:])
    print(f"[extract] query={len(q['name'])} gallery={len(g['name'])} dim={feats.shape[1]} "
          f"({time.time()-t0:.0f}s)", flush=True)
    np.savez(cli.cache_feat,
             q_feat=q['feat'], q_pid=q['pid'], q_cam=q['cam'], q_name=q['name'],
             g_feat=g['feat'], g_pid=g['pid'], g_cam=g['cam'], g_name=g['name'])
    print(f"[extract] cached -> {cli.cache_feat}", flush=True)
    return q, g


# =========================================================================== #
# 2. EVAL  (market protocol: drop same pid&cam junk; also drop pid==-1 junk gallery)
# =========================================================================== #
def eval_map(distmat, q_pid, q_cam, g_pid, g_cam, max_rank=20):
    num_q = distmat.shape[0]
    indices = np.argsort(distmat, axis=1)
    all_AP, all_cmc = [], []
    nvalid = 0
    for i in range(num_q):
        order = indices[i]
        remove = (g_pid[order] == q_pid[i]) & (g_cam[order] == q_cam[i])
        keep = ~remove
        m = (g_pid[order][keep] == q_pid[i]).astype(np.int32)
        if not m.any():
            continue
        nvalid += 1
        cmc = m.cumsum(); cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        tmp = m.cumsum()
        prec = tmp / (np.arange(len(m)) + 1.0)
        all_AP.append((prec * m).sum() / m.sum())
    all_cmc = np.asarray(all_cmc).mean(0)
    return dict(mAP=float(np.mean(all_AP)) * 100, r1=float(all_cmc[0]) * 100,
                r5=float(all_cmc[4]) * 100, r10=float(all_cmc[9]) * 100, nq=nvalid)


def per_query_ap(distmat, q_pid, q_cam, g_pid, g_cam):
    num_q = distmat.shape[0]
    indices = np.argsort(distmat, axis=1)
    aps = np.full(num_q, -1.0)
    for i in range(num_q):
        order = indices[i]
        keep = ~((g_pid[order] == q_pid[i]) & (g_cam[order] == q_cam[i]))
        m = (g_pid[order][keep] == q_pid[i]).astype(np.int32)
        if not m.any():
            continue
        prec = m.cumsum() / (np.arange(len(m)) + 1.0)
        aps[i] = (prec * m).sum() / m.sum()
    return aps


# =========================================================================== #
# 3. HUBNESS CORE
# =========================================================================== #
def topk_per_query(sim, k):
    """Return (num_q, k) indices of the top-k MOST SIMILAR gallery per query.

    NOTE: This is the RAW retrieval kNN (no junk removal), matching the design's
    directed kNN-graph definition: a gallery item is a hub if it lands in many
    queries' top-k regardless of the eval junk rule. (We separately verify the
    intervention with the full eval protocol's junk removal.)"""
    # argpartition top-k then sort within
    idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(sim.shape[0])[:, None]
    order = np.argsort(-sim[rows, idx], axis=1)
    return idx[rows, order]


def compute_Hk(sim, q_pid, g_pid, k, signed='neg'):
    """H_k(g): how many queries put g in their top-k.
       signed='neg' -> only count queries with y_q != y_g  (NEGATIVE in-degree)
       signed='all' -> count every query                    (TOTAL  in-degree)
       signed='pos' -> only count queries with y_q == y_g   (POSITIVE in-degree)
    Returns H (num_g,) int and the per-query top-k index matrix (num_q,k)."""
    tk = topk_per_query(sim, k)                       # (num_q, k)
    H = np.zeros(sim.shape[1], dtype=np.int64)
    qp = q_pid[:, None]                               # (num_q,1)
    for col in range(k):
        gj = tk[:, col]
        if signed == 'all':
            np.add.at(H, gj, 1)
        else:
            same = (g_pid[gj] == q_pid)
            if signed == 'neg':
                sel = ~same
            else:  # 'pos'
                sel = same
            np.add.at(H, gj[sel], 1)
    return H, tk


def query_hub_mass(tk, H, q_pid, g_pid):
    """M(q) = sum over g in top-k(q) with y_g != y_q of H[g]  (negative hub mass)."""
    M = np.zeros(tk.shape[0], dtype=np.float64)
    for col in range(tk.shape[1]):
        gj = tk[:, col]
        neg = (g_pid[gj] != q_pid)
        M += np.where(neg, H[gj], 0.0)
    return M


# =========================================================================== #
# 4. STATS helpers (no scipy)
# =========================================================================== #
def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float('nan'), 0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    rho = float((rx * ry).sum() / denom) if denom > 0 else float('nan')
    return rho, len(x)


def partial_spearman(x, y, Z):
    """Spearman partial corr of x,y controlling for one or more covariates Z
    (Z: (n,) or (n,m)). Correlate rank-residuals of x|Z and y|Z."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(Z).all(axis=1)
    x, y, Z = x[ok], y[ok], Z[ok]
    if len(x) < 5:
        return float('nan'), 0
    def rank(v):
        return np.argsort(np.argsort(v)).astype(float)
    rx, ry = rank(x), rank(y)
    Zr = np.column_stack([np.ones(len(x))] + [rank(Z[:, j]) for j in range(Z.shape[1])])
    def resid(r):
        beta, *_ = np.linalg.lstsq(Zr, r, rcond=None)
        return r - Zr @ beta
    ex, ey = resid(rx), resid(ry)
    denom = np.sqrt((ex**2).sum() * (ey**2).sum())
    rho = float((ex * ey).sum() / denom) if denom > 0 else float('nan')
    return rho, len(x)


def perm_pvalue(x, y, rho_obs, n_perm=1000):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5 or not np.isfinite(rho_obs):
        return float('nan')
    cnt = 0
    for _ in range(n_perm):
        r, _ = spearman(x, RNG.permutation(y))
        if abs(r) >= abs(rho_obs):
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


# =========================================================================== #
# MAIN
# =========================================================================== #
def main():
    DS = cli.dataset
    print("#" * 80)
    print(f"# GALLERY HUBNESS KILL-SWITCH  dataset={DS}  ckpt={cli.ckpt}")
    print("#" * 80)

    # ---- features ----
    if cli.reuse_feat and os.path.exists(cli.cache_feat):
        z = np.load(cli.cache_feat, allow_pickle=True)
        q = dict(feat=z['q_feat'], pid=z['q_pid'], cam=z['q_cam'], name=z['q_name'])
        g = dict(feat=z['g_feat'], pid=z['g_pid'], cam=z['g_cam'], name=z['g_name'])
        print(f"[reuse] features from {cli.cache_feat}: q={len(q['name'])} g={len(g['name'])}")
    else:
        q, g = extract_features()

    # ---- drop junk gallery (pid == -1, market distractors) ----
    keep_g = g['pid'] != -1
    for key in ('feat', 'pid', 'cam', 'name'):
        g[key] = g[key][keep_g]
    # smoke cap
    if cli.smoke > 0:
        for key in ('feat', 'pid', 'cam', 'name'):
            q[key] = q[key][:cli.smoke]
        print(f"[SMOKE] capped query -> {len(q['name'])}")

    qf = q['feat'].astype(np.float32); gf = g['feat'].astype(np.float32)
    qf /= (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
    gf /= (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
    q_pid, q_cam = q['pid'], q['cam']
    g_pid, g_cam = g['pid'], g['cam']
    Nq, Ng = qf.shape[0], gf.shape[0]
    print(f"[data] Nq={Nq} Ng={Ng} dim={qf.shape[1]}  "
          f"#q-pids={len(np.unique(q_pid))} #g-pids={len(np.unique(g_pid))}")

    # cosine SIMILARITY (higher = closer) and DISTANCE (1 - sim)
    sim = qf @ gf.T                 # (Nq,Ng)
    dm = 1.0 - sim

    # ---- sanity mAP ----
    res = eval_map(dm, q_pid, q_cam, g_pid, g_cam)
    print(f"\n[SANITY] frozen cosine (global feat) mAP={res['mAP']:.2f} R1={res['r1']:.2f} "
          f"R5={res['r5']:.2f} R10={res['r10']:.2f} nq={res['nq']}  "
          f"(exp260b equal_concat ref mAP 94.4; this is the GLOBAL-branch single vector, "
          f"expected somewhat lower)")

    # ======================================================================= #
    # per-query AP and "false hits" under the EVAL protocol (junk removed)
    # ======================================================================= #
    aps = per_query_ap(dm, q_pid, q_cam, g_pid, g_cam)
    # top-1 / top-10 gallery indices AFTER junk removal, and whether they are wrong.
    indices = np.argsort(dm, axis=1)
    false_top1_g = []    # gallery indices that are a FALSE rank-1
    false_top10_g = []   # gallery indices that are FALSE within rank-10
    for i in range(Nq):
        order = indices[i]
        keep = ~((g_pid[order] == q_pid[i]) & (g_cam[order] == q_cam[i]))
        order_k = order[keep]
        # is there any valid positive? (else this query is dropped from eval)
        if not ((g_pid[order_k] == q_pid[i]).any()):
            continue
        top1 = order_k[0]
        if g_pid[top1] != q_pid[i]:
            false_top1_g.append(top1)
        top10 = order_k[:10]
        wrong10 = top10[g_pid[top10] != q_pid[i]]
        false_top10_g.extend(wrong10.tolist())
    false_top1_g = np.asarray(false_top1_g, dtype=np.int64)
    false_top10_g = np.asarray(false_top10_g, dtype=np.int64)
    print(f"[eval] #false-top1 hits={len(false_top1_g)}  #false-top10 hits={len(false_top10_g)}")

    # ======================================================================= #
    # build H_k (NEG / ALL / POS in-degree) for each k; M(q) at k_main
    # ======================================================================= #
    print("\n" + "=" * 80)
    print("H_k construction (NEGATIVE in-degree = cross-ID mis-attraction)")
    print("=" * 80)
    H_neg, H_all, H_pos, TK = {}, {}, {}, {}
    for k in cli.ks:
        H_neg[k], TK[k] = compute_Hk(sim, q_pid, g_pid, k, signed='neg')
        H_all[k], _ = compute_Hk(sim, q_pid, g_pid, k, signed='all')
        H_pos[k], _ = compute_Hk(sim, q_pid, g_pid, k, signed='pos')
        hn = H_neg[k]
        print(f"  k={k:2d}: H_neg max={hn.max():4d} mean={hn.mean():.3f}  "
              f"#gallery with H_neg>0 = {(hn>0).sum()} ({100*(hn>0).mean():.1f}%)  "
              f"top-1% H_neg threshold = {np.quantile(hn, 0.99):.0f}")
    km = cli.k_main
    Hk = H_neg[km]
    M = query_hub_mass(TK[km], Hk, q_pid, g_pid)

    # ======================================================================= #
    # CORE TEST 1 — hub concentration
    # ======================================================================= #
    print("\n" + "=" * 80)
    print("CORE TEST 1 — hub concentration (top-1% high-H_k gallery absorb what % of false hits)")
    print("=" * 80)
    for k in cli.ks:
        hn = H_neg[k]
        thr = np.quantile(hn, 0.99)
        # top-1% set: the highest-H_neg gallery (break ties by including >= threshold,
        # then trim to ~1% by taking the largest). Use a rank cut for an exact 1%.
        order_h = np.argsort(-hn)
        n_top = max(1, int(round(0.01 * Ng)))
        top1pct = set(order_h[:n_top].tolist())
        # share of false-top1 / false-top10 hits whose gallery is in the top-1% hub set
        if len(false_top1_g):
            sh1 = float(np.mean([gj in top1pct for gj in false_top1_g])) * 100
        else:
            sh1 = float('nan')
        if len(false_top10_g):
            sh10 = float(np.mean([gj in top1pct for gj in false_top10_g])) * 100
        else:
            sh10 = float('nan')
        # also: what fraction of ALL gallery that ever appear as a false hit?
        print(f"  k={k:2d}: top-1% hub gallery (n={n_top}, H_neg>={hn[order_h[n_top-1]]}) "
              f"absorb  false-top1 {sh1:5.1f}%   false-top10 {sh10:5.1f}%   "
              f"(uniform expectation = 1.0%)")

    # ======================================================================= #
    # CORE TEST 2 — M(q) explains per-query AP error vs cheap proxies
    # ======================================================================= #
    print("\n" + "=" * 80)
    print("CORE TEST 2 — M(q) vs per-query AP error, against cheap difficulty proxies")
    print("=" * 80)
    err = 1.0 - aps           # AP error (higher = worse)
    valid = aps >= 0
    # cheap proxies
    qnorm = np.linalg.norm(q['feat'][valid].astype(np.float64), axis=1)   # raw (pre-renorm) norm
    # top1 margin: sim(top1) - sim(top2) under the FULL kNN (retrieval geometry)
    sim_sorted = np.sort(sim, axis=1)
    top1margin = sim_sorted[:, -1] - sim_sorted[:, -2]
    # camera-pair: query camera id (categorical) — use as a covariate via its mean err? we
    # use n_cams_in_top as a proxy; but for partial corr we need a scalar. Use the camera
    # id directly as ordinal covariate AND an "#distinct gallery cams in top-k" measure.
    # #gallery positives for this query (valid same-pid, diff-cam)
    n_pos = np.array([((g_pid == q_pid[i]) & (g_cam != q_cam[i])).sum() for i in range(Nq)], float)
    # camera-pair severity proxy: fraction of top-k that share the query camera (junk-prone)
    same_cam_frac = np.zeros(Nq)
    for col in range(km):
        gj = TK[km][:, col]
        same_cam_frac += (g_cam[gj] == q_cam)
    same_cam_frac /= km

    def rep(name, x):
        rho, n = spearman(M[valid], x[valid]) if name == '__M__' else spearman(err[valid], x[valid])
        return rho, n

    rM, nM = spearman(err[valid], M[valid])
    rN, _ = spearman(err[valid], -np.linalg.norm(q['feat'].astype(np.float64), axis=1)[valid])  # low norm -> high err?
    rNorm_raw, _ = spearman(err[valid], np.linalg.norm(q['feat'].astype(np.float64), axis=1)[valid])
    rMar, _ = spearman(err[valid], -top1margin[valid])    # small margin -> high err
    rNpos, _ = spearman(err[valid], -n_pos[valid])        # few positives -> high err
    rCam, _ = spearman(err[valid], same_cam_frac[valid])  # more same-cam in topk -> high err
    print(f"  rho(AP-error, M(q))                 = {rM:+.4f}  (n={nM})  [HUB MASS]")
    print(f"  rho(AP-error, feature-norm)         = {rNorm_raw:+.4f}")
    print(f"  rho(AP-error, -top1-margin)         = {rMar:+.4f}")
    print(f"  rho(AP-error, -#gallery-positives)  = {rNpos:+.4f}")
    print(f"  rho(AP-error, same-cam-frac-in-topk)= {rCam:+.4f}")
    pM = perm_pvalue(err[valid], M[valid], rM, n_perm=1000)
    print(f"  perm-p(M(q)) = {pM:.4f}")

    # partial: M(q) controlling EACH proxy, and ALL proxies jointly
    feat_norm = np.linalg.norm(q['feat'].astype(np.float64), axis=1)
    cov_stack = np.column_stack([feat_norm[valid], top1margin[valid], n_pos[valid], same_cam_frac[valid]])
    pr_norm, _ = partial_spearman(err[valid], M[valid], feat_norm[valid])
    pr_mar, _ = partial_spearman(err[valid], M[valid], top1margin[valid])
    pr_cam, _ = partial_spearman(err[valid], M[valid], same_cam_frac[valid])
    pr_npos, _ = partial_spearman(err[valid], M[valid], n_pos[valid])
    pr_all, nall = partial_spearman(err[valid], M[valid], cov_stack)
    print(f"  partial rho(AP-error, M | norm)            = {pr_norm:+.4f}")
    print(f"  partial rho(AP-error, M | top1-margin)     = {pr_mar:+.4f}")
    print(f"  partial rho(AP-error, M | same-cam-frac)   = {pr_cam:+.4f}")
    print(f"  partial rho(AP-error, M | #gallery-pos)    = {pr_npos:+.4f}")
    print(f"  partial rho(AP-error, M | ALL 4 proxies)   = {pr_all:+.4f}  (n={nall})  "
          f"[D3 line: must stay clearly nonzero]")

    # ======================================================================= #
    # CORE TEST 3 — zero-training intervention score' = cos - lambda*log(1+H_k)
    # ======================================================================= #
    print("\n" + "=" * 80)
    print(f"CORE TEST 3 — intervention score'(q,g) = cosine - lambda*log(1+H_k(g)), k={km}")
    print("=" * 80)
    logH = np.log1p(Hk.astype(np.float64))[None, :]    # (1,Ng)
    base = eval_map(dm, q_pid, q_cam, g_pid, g_cam)
    print(f"  lambda=0 (baseline): mAP={base['mAP']:.3f} R1={base['r1']:.3f} R5={base['r5']:.3f}")
    best = dict(lam=0.0, mAP=base['mAP'], r1=base['r1'])
    intervened_dm_best = dm
    for lam in cli.lambdas:
        if lam == 0.0:
            continue
        score = sim - lam * logH          # higher = better
        dmi = -score                      # distance for argsort
        r = eval_map(dmi, q_pid, q_cam, g_pid, g_cam)
        flag = ''
        if r['mAP'] > best['mAP']:
            best = dict(lam=lam, mAP=r['mAP'], r1=r['r1']); intervened_dm_best = dmi
            flag = '  <== best so far'
        print(f"  lambda={lam:<6.3f}: mAP={r['mAP']:.3f} (d{r['mAP']-base['mAP']:+.3f})  "
              f"R1={r['r1']:.3f} (d{r['r1']-base['r1']:+.3f})  R5={r['r5']:.3f}{flag}")
    print(f"  >> BEST intervention: lambda={best['lam']} mAP={best['mAP']:.3f} "
          f"(d{best['mAP']-base['mAP']:+.3f}) R1={best['r1']:.3f} (d{best['r1']-base['r1']:+.3f})")

    # gain concentration: does the per-query AP gain concentrate on high-M(q)?
    aps_base = per_query_ap(dm, q_pid, q_cam, g_pid, g_cam)
    aps_int = per_query_ap(intervened_dm_best, q_pid, q_cam, g_pid, g_cam)
    dgain = aps_int - aps_base
    sel = (aps_base >= 0) & (aps_int >= 0)
    rho_gain, _ = spearman(M[sel], dgain[sel])
    # bucket by M(q) quartile
    order = np.argsort(M[sel]); qb = np.array_split(order, 4)
    print(f"\n  gain concentration (best lambda={best['lam']}):")
    print(f"    Spearman(M(q), AP-gain) = {rho_gain:+.4f}  (expect POSITIVE: high-M queries gain more)")
    Msel = M[sel]; dgsel = dgain[sel]
    for b, idxs in enumerate(qb):
        print(f"    M-quartile Q{b} (n={len(idxs):4d}, mean M={Msel[idxs].mean():9.1f}): "
              f"mean AP-gain = {100*dgsel[idxs].mean():+.3f}")

    # ======================================================================= #
    # DESTRUCTIVE CONTROLS
    # ======================================================================= #
    print("\n" + "#" * 80)
    print("DESTRUCTIVE CONTROLS (decide novelty)")
    print("#" * 80)

    # ---- D1: shuffle H_k -> intervention gain must vanish ----
    print("\n-- D1: shuffle H_k (permute gallery H values) -> gain must vanish --")
    Hk_sh = RNG.permutation(Hk)
    logH_sh = np.log1p(Hk_sh.astype(np.float64))[None, :]
    d1_best = base['mAP']
    for lam in cli.lambdas:
        if lam == 0.0:
            continue
        r = eval_map(-(sim - lam * logH_sh), q_pid, q_cam, g_pid, g_cam)
        d1_best = max(d1_best, r['mAP'])
    print(f"  best mAP with SHUFFLED H_k over lambda sweep = {d1_best:.3f}  "
          f"(baseline {base['mAP']:.3f}; real-best {best['mAP']:.3f}).  "
          f"D1 PASS if shuffled gain ~ 0 (<< real gain)")

    # ---- D2: camera-correction equivalence ----
    print("\n-- D2: camera-correction equivalence (hub must beat pure camera fixes) --")
    # D2a: same-camera down-weight. Push gallery items sharing the query camera further.
    #      score = cos - gamma * [g_cam == q_cam].  This is the cheapest camera fix.
    same_cam_mat = (g_cam[None, :] == q_cam[:, None]).astype(np.float64)   # (Nq,Ng)
    d2a_best = dict(g=0.0, mAP=base['mAP'])
    for gamma in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        r = eval_map(-(sim - gamma * same_cam_mat), q_pid, q_cam, g_pid, g_cam)
        if r['mAP'] > d2a_best['mAP']:
            d2a_best = dict(g=gamma, mAP=r['mAP'])
    print(f"  D2a same-camera down-weight  best mAP = {d2a_best['mAP']:.3f} "
          f"(gamma={d2a_best['g']}, d{d2a_best['mAP']-base['mAP']:+.3f})")

    # D2b: CA-Jaccard-style camera-aware k-reciprocal re-rank (camera-only correction).
    #      A lightweight k-reciprocal re-ranking computed in camera-blocked fashion:
    #      Jaccard distance of expanded reciprocal neighbor sets, with the original
    #      camera-pair distances unchanged (standard k-reciprocal as the camera-agnostic
    #      strong test-time baseline) AND a camera-aware variant that removes same-cam
    #      gallery from the reciprocal sets (CA-Jaccard spirit).
    print("  D2b k-reciprocal re-rank (strong test-time baseline) and CA variant:")
    rr = kreciprocal_rerank(qf, gf, q_cam, g_cam, k1=20, k2=6, lam=0.3, camera_aware=False)
    r_rr = eval_map(rr, q_pid, q_cam, g_pid, g_cam)
    rr_ca = kreciprocal_rerank(qf, gf, q_cam, g_cam, k1=20, k2=6, lam=0.3, camera_aware=True)
    r_rr_ca = eval_map(rr_ca, q_pid, q_cam, g_pid, g_cam)
    print(f"    k-reciprocal (plain)      mAP = {r_rr['mAP']:.3f}  R1={r_rr['r1']:.3f} "
          f"(d{r_rr['mAP']-base['mAP']:+.3f})")
    print(f"    k-reciprocal (camera-aware) mAP = {r_rr_ca['mAP']:.3f}  R1={r_rr_ca['r1']:.3f} "
          f"(d{r_rr_ca['mAP']-base['mAP']:+.3f})")
    print(f"  >> D2 verdict: hub intervention gain (d{best['mAP']-base['mAP']:+.3f}) vs "
          f"camera-only fixes (D2a d{d2a_best['mAP']-base['mAP']:+.3f}). If hub <= camera fix, "
          f"NO independent value.")

    # ---- D3: already reported (partial rho(AP-error, M | norm+margin+camera+npos)) ----
    print("\n-- D3: M(q) partial corr controlling norm+margin+camera+#pos (from Test 2) --")
    print(f"  partial rho(AP-error, M | ALL 4 proxies) = {pr_all:+.4f}  "
          f"(D3 PASS if clearly nonzero, i.e. M not reducible to cheap difficulty)")

    # ---- D4: NEGATIVE vs ALL (and POS) in-degree ----
    print("\n-- D4 (CRITICAL): NEGATIVE in-degree vs ALL / POS in-degree --")
    # Build interventions using H_all and H_pos at k_main; compare gain & correlation.
    def best_intervention(Hvec):
        logh = np.log1p(Hvec.astype(np.float64))[None, :]
        bm = base['mAP']; bl = 0.0; bdm = dm
        for lam in cli.lambdas:
            if lam == 0.0:
                continue
            r = eval_map(-(sim - lam * logh), q_pid, q_cam, g_pid, g_cam)
            if r['mAP'] > bm:
                bm = r['mAP']; bl = lam; bdm = -(sim - lam * logh)
        return bm, bl, bdm
    Hk_all = H_all[km]; Hk_pos = H_pos[km]
    bm_neg, bl_neg = best['mAP'], best['lam']
    bm_all, bl_all, _ = best_intervention(Hk_all)
    bm_pos, bl_pos, _ = best_intervention(Hk_pos)
    print(f"  intervention with NEG in-degree: best mAP={bm_neg:.3f} (d{bm_neg-base['mAP']:+.3f}, lam={bl_neg})")
    print(f"  intervention with ALL in-degree: best mAP={bm_all:.3f} (d{bm_all-base['mAP']:+.3f}, lam={bl_all})")
    print(f"  intervention with POS in-degree: best mAP={bm_pos:.3f} (d{bm_pos-base['mAP']:+.3f}, lam={bl_pos})")
    # correlation of AP-error with the negative vs total in-degree mass
    M_all = query_hub_mass_signed(TK[km], Hk_all, q_pid, g_pid, signed='all')
    rho_Mneg, _ = spearman(err[valid], M[valid])
    rho_Mall, _ = spearman(err[valid], M_all[valid])
    print(f"  rho(AP-error, M_neg)={rho_Mneg:+.4f}   rho(AP-error, M_all)={rho_Mall:+.4f}")
    # how correlated are NEG and ALL in-degree across gallery? (if ~1, they are the same)
    rHH, _ = spearman(Hk.astype(float), Hk_all.astype(float))
    print(f"  Spearman(H_neg, H_all) across gallery = {rHH:+.4f}  "
          f"(if ~1.0 the negative/total distinction is moot -> D4 weak)")
    print(f"  >> D4 verdict: NEG must drive the gain; if ALL in-degree gives the SAME gain "
          f"and H_neg~H_all, the signal is just 'popular sample', not cross-ID mis-attraction.")

    # ======================================================================= #
    # FINAL SUMMARY
    # ======================================================================= #
    print("\n" + "#" * 80)
    print(f"SUMMARY / VERDICT  ({DS})")
    print("#" * 80)
    # recompute T1 share at k_main for the headline
    hn = H_neg[km]; order_h = np.argsort(-hn); n_top = max(1, int(round(0.01 * Ng)))
    top1pct = set(order_h[:n_top].tolist())
    sh1 = float(np.mean([gj in top1pct for gj in false_top1_g])) * 100 if len(false_top1_g) else float('nan')
    sh10 = float(np.mean([gj in top1pct for gj in false_top10_g])) * 100 if len(false_top10_g) else float('nan')
    print(f"  sanity frozen mAP                         = {res['mAP']:.2f}")
    print(f"  T1 top-1% hub absorb false-top1/top10     = {sh1:.1f}% / {sh10:.1f}%  "
          f"(want >=20-30%, uniform=1%)")
    print(f"  T2 rho(AP-error, M)                       = {rM:+.4f}  (perm-p {pM:.4f})")
    print(f"  T2 vs norm/margin/cam/npos                = "
          f"{rNorm_raw:+.3f}/{rMar:+.3f}/{rCam:+.3f}/{rNpos:+.3f}")
    print(f"  T3 best intervention mAP gain             = {best['mAP']-base['mAP']:+.3f} "
          f"(lambda={best['lam']})  gain-concentration rho={rho_gain:+.3f}")
    print(f"  D1 shuffled-H gain                        = {d1_best-base['mAP']:+.3f} (want ~0)")
    print(f"  D2 camera-only best gain (downweight/RR)  = {d2a_best['mAP']-base['mAP']:+.3f} / "
          f"{r_rr['mAP']-base['mAP']:+.3f}")
    print(f"  D3 partial(AP-error, M | 4 proxies)       = {pr_all:+.4f} (want clearly !=0)")
    print(f"  D4 NEG/ALL/POS intervention gain          = "
          f"{bm_neg-base['mAP']:+.3f}/{bm_all-base['mAP']:+.3f}/{bm_pos-base['mAP']:+.3f}; "
          f"Spearman(H_neg,H_all)={rHH:+.3f}")
    print("\n[done] hubness kill-switch complete.")


def query_hub_mass_signed(tk, H, q_pid, g_pid, signed='all'):
    """M(q) summed over top-k neighbors. signed='all' sums H over ALL neighbors;
    signed='neg' only over cross-ID neighbors."""
    M = np.zeros(tk.shape[0], dtype=np.float64)
    for col in range(tk.shape[1]):
        gj = tk[:, col]
        if signed == 'all':
            M += H[gj]
        else:
            neg = (g_pid[gj] != q_pid)
            M += np.where(neg, H[gj], 0.0)
    return M


# --------------------------------------------------------------------------- #
# k-reciprocal re-ranking (Zhong et al. 2017), numpy, with optional camera-aware
# removal of same-camera gallery from the reciprocal neighbor sets (CA-Jaccard
# spirit). Returns a (Nq,Ng) re-ranked DISTANCE matrix.
# --------------------------------------------------------------------------- #
def kreciprocal_rerank(qf, gf, q_cam, g_cam, k1=20, k2=6, lam=0.3, camera_aware=False):
    qf = qf.astype(np.float32); gf = gf.astype(np.float32)
    Nq, Ng = qf.shape[0], gf.shape[0]
    allf = np.concatenate([qf, gf], 0)            # (Nq+Ng, D)
    cams = np.concatenate([q_cam, g_cam], 0)
    # original distance = 1 - cosine (features are L2-normed)
    orig = 2.0 - 2.0 * (allf @ allf.T)            # squared-Euclidean proxy on unit sphere
    orig = np.maximum(orig, 0.0)
    N = Nq + Ng
    initial_rank = np.argsort(orig, axis=1).astype(np.int32)

    V = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        # k-reciprocal neighbors
        fwd = initial_rank[i, :k1 + 1]
        bwd_k = k1 + 1
        recip = []
        for cand in fwd:
            cand_fwd = initial_rank[cand, :bwd_k]
            if i in cand_fwd:
                recip.append(cand)
        recip = np.array(recip, dtype=np.int32) if len(recip) else np.array([i], dtype=np.int32)
        # expand (half-neighbors)
        recip_exp = list(recip)
        for cand in recip:
            cand_half = initial_rank[cand, :int(np.around(k1 / 2.0)) + 1]
            if len(np.intersect1d(cand_half, recip)) > 2.0 / 3.0 * len(cand_half):
                recip_exp.extend(cand_half.tolist())
        recip_exp = np.unique(np.array(recip_exp, dtype=np.int32))
        if camera_aware:
            # CA-Jaccard spirit: drop same-camera items (except self) from the set so
            # the reciprocal neighborhood is built across cameras.
            recip_exp = recip_exp[(cams[recip_exp] != cams[i]) | (recip_exp == i)]
            if len(recip_exp) == 0:
                recip_exp = np.array([i], dtype=np.int32)
        w = np.exp(-orig[i, recip_exp])
        V[i, recip_exp] = (w / w.sum()).astype(np.float32)

    # local query expansion (k2)
    if k2 > 1:
        V_qe = np.zeros_like(V)
        for i in range(N):
            V_qe[i] = V[initial_rank[i, :k2]].mean(0)
        V = V_qe

    # Jaccard distance
    invIndex = [np.where(V[:, j] != 0)[0] for j in range(N)]
    jaccard = np.zeros((Nq, Ng), dtype=np.float32)
    for i in range(Nq):
        nz = np.where(V[i] != 0)[0]
        minsum = np.zeros(N, dtype=np.float32)
        for j in nz:
            cols = invIndex[j]
            minsum[cols] += np.minimum(V[i, j], V[cols, j])
        jd = 1.0 - minsum / (2.0 - minsum + 1e-12)
        jaccard[i] = jd[Nq:]                     # gallery columns
    final = (1.0 - lam) * jaccard + lam * orig[:Nq, Nq:]
    return final


if __name__ == '__main__':
    main()
