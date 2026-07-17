#!/usr/bin/env python3
"""Hubness diagnostic VERIFICATION — P0 (statistical closure) + P4 (k-reciprocal reframe).

ZERO-TRAINING: cached frozen features (.npz) + numpy only. No model, no backward.

P0 (CRITICAL — does the diagnostic survive de-circularization?):
  The headline rho(AP-error, M(q)) ~ +0.60 may be CIRCULAR because M(q) sums H_k(g)
  over g in q's OWN top-k, and H_k(g) itself counts q's OWN contribution. We break this
  three ways and report whether rho survives:

  P0a leave-one-query-out (LOO): when summing M(q), use H_k^{-q}(g) = H_k(g) minus the
      (0/1) contribution that query q itself made to g's in-degree. This removes the
      direct self-loop. Report rho(AP-err, M_loo).

  P0b held-out split: estimate H_k on HALF the queries (the "estimation" split), then for
      the OTHER half compute M(q) from that held-out H_k and correlate with their AP error.
      No query ever sees its own contribution. Report rho + bootstrap 95% CI + permutation p.
      (Averaged over several random splits for stability; primary number = single seeded split.)

  P0c stronger cheap controls: partial corr of M(q) | {#false-in-topk, topk-precision,
      first-positive-rank, mean-negative-similarity, top1-correct} (a much harder baseline
      than the original norm/margin/cam/#pos). Does M still add signal?

P4 (REFRAME — is M(q) exactly what k-reciprocal repairs?):
  per-query k-reciprocal gain = AP_rerank(q) - AP_base(q). Report rho(M(q), gain) and a
  binned trend. If high-M queries are precisely the ones k-reciprocal fixes most, the paper
  value becomes "we explain WHY a standard tool works".

Run on lab-3090-d:
  /root/miniconda3/envs/solider-reid/bin/python \
    experiments/cargo_cvpb/hub_verify_p0_p4.py \
    --cache_feat /tmp/hub_oduke_feats.npz --dataset occluded_duke 2>&1 | tee /tmp/hub_p0p4_oduke.log
  ... --cache_feat /tmp/hub_market_feats.npz --dataset market1501 ...
"""
import os, sys, time, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--cache_feat', required=True)
ap.add_argument('--dataset', default='occluded_duke')
ap.add_argument('--k_main', type=int, default=10)
ap.add_argument('--ks', type=int, nargs='+', default=[5, 10, 20])
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--n_boot', type=int, default=2000)
ap.add_argument('--n_perm', type=int, default=2000)
ap.add_argument('--n_splits', type=int, default=20, help='#random held-out splits to average P0b')
ap.add_argument('--rr_k1', type=int, default=20)
ap.add_argument('--rr_k2', type=int, default=6)
ap.add_argument('--rr_lam', type=float, default=0.3)
cli = ap.parse_args()
RNG = np.random.RandomState(cli.seed)


# ============================================================ stats (no scipy)
def _rank(v):
    return np.argsort(np.argsort(v)).astype(float)

def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float('nan'), 0
    rx, ry = _rank(x), _rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return (float((rx * ry).sum() / denom) if denom > 0 else float('nan')), len(x)

def partial_spearman(x, y, Z):
    """partial Spearman of x,y controlling covariates Z (n,) or (n,m)."""
    x = np.asarray(x, float); y = np.asarray(y, float); Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(Z).all(axis=1)
    x, y, Z = x[ok], y[ok], Z[ok]
    if len(x) < 5:
        return float('nan'), 0
    rx, ry = _rank(x), _rank(y)
    Zr = np.column_stack([np.ones(len(x))] + [_rank(Z[:, j]) for j in range(Z.shape[1])])
    def resid(r):
        beta, *_ = np.linalg.lstsq(Zr, r, rcond=None)
        return r - Zr @ beta
    ex, ey = resid(rx), resid(ry)
    denom = np.sqrt((ex**2).sum() * (ey**2).sum())
    return (float((ex * ey).sum() / denom) if denom > 0 else float('nan')), len(x)

def perm_pvalue(x, y, rho_obs, n_perm, rng):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5 or not np.isfinite(rho_obs):
        return float('nan')
    cnt = 0
    for _ in range(n_perm):
        r, _ = spearman(x, rng.permutation(y))
        if abs(r) >= abs(rho_obs):
            cnt += 1
    return (cnt + 1) / (n_perm + 1)

def boot_ci_spearman(x, y, n_boot, rng, alpha=0.05):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 5:
        return float('nan'), float('nan')
    rhos = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        r, _ = spearman(x[idx], y[idx])
        rhos[b] = r
    rhos = rhos[np.isfinite(rhos)]
    lo = float(np.percentile(rhos, 100 * alpha / 2))
    hi = float(np.percentile(rhos, 100 * (1 - alpha / 2)))
    return lo, hi


# ============================================================ retrieval core
def per_query_ap_and_topk(sim, q_pid, q_cam, g_pid, g_cam, k_main):
    """Single pass: per-query AP (junk removed) + RAW top-k indices (no junk removal,
    matching the kNN-graph H_k definition) + diagnostic per-query features."""
    Nq = sim.shape[0]
    aps = np.full(Nq, -1.0)
    tk_raw = np.zeros((Nq, k_main), dtype=np.int64)
    n_false_topk = np.zeros(Nq)           # #different-id within RAW top-k
    topk_prec = np.zeros(Nq)              # fraction same-id within EVAL top-k (junk removed)
    first_pos_rank = np.full(Nq, np.nan)  # rank of first true positive (eval order)
    mean_neg_sim = np.full(Nq, np.nan)    # mean cosine to different-id in RAW top-k
    top1_correct = np.zeros(Nq)           # eval top-1 is correct (1/0)
    same_cam_frac = np.zeros(Nq)          # frac of RAW top-k sharing q camera
    # full argsort once (distance asc == sim desc)
    order_all = np.argsort(-sim, axis=1)
    for i in range(Nq):
        oa = order_all[i]
        # raw top-k (kNN graph, no junk removal)
        tk = oa[:k_main]
        tk_raw[i] = tk
        neg_mask_raw = g_pid[tk] != q_pid[i]
        n_false_topk[i] = neg_mask_raw.sum()
        if neg_mask_raw.any():
            mean_neg_sim[i] = sim[i, tk[neg_mask_raw]].mean()
        same_cam_frac[i] = (g_cam[tk] == q_cam[i]).mean()
        # eval order (junk removed)
        keep = ~((g_pid[oa] == q_pid[i]) & (g_cam[oa] == q_cam[i]))
        oe = oa[keep]
        m = (g_pid[oe] == q_pid[i]).astype(np.int32)
        if not m.any():
            continue
        prec = m.cumsum() / (np.arange(len(m)) + 1.0)
        aps[i] = (prec * m).sum() / m.sum()
        topk_prec[i] = m[:k_main].mean()
        first_pos_rank[i] = int(np.argmax(m == 1)) + 1
        top1_correct[i] = float(m[0] == 1)
    return dict(aps=aps, tk_raw=tk_raw, n_false_topk=n_false_topk, topk_prec=topk_prec,
                first_pos_rank=first_pos_rank, mean_neg_sim=mean_neg_sim,
                top1_correct=top1_correct, same_cam_frac=same_cam_frac)


def eval_map(distmat, q_pid, q_cam, g_pid, g_cam, max_rank=20):
    num_q = distmat.shape[0]
    indices = np.argsort(distmat, axis=1)
    all_AP, all_cmc = [], []
    for i in range(num_q):
        order = indices[i]
        keep = ~((g_pid[order] == q_pid[i]) & (g_cam[order] == q_cam[i]))
        m = (g_pid[order][keep] == q_pid[i]).astype(np.int32)
        if not m.any():
            continue
        cmc = m.cumsum(); cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        prec = m.cumsum() / (np.arange(len(m)) + 1.0)
        all_AP.append((prec * m).sum() / m.sum())
    all_cmc = np.asarray(all_cmc).mean(0)
    return dict(mAP=float(np.mean(all_AP)) * 100, r1=float(all_cmc[0]) * 100, nq=len(all_AP))


def per_query_ap_from_dist(distmat, q_pid, q_cam, g_pid, g_cam):
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


# ============================================================ H_k with attribution
def compute_Hk_neg_with_attrib(tk_raw, q_pid, g_pid, Ng):
    """H_k(g) NEGATIVE in-degree + per (query,slot) attribution so we can subtract a
    single query's contribution (LOO).
    Returns H (Ng,), and contributes[i] = set/array of gallery idx that query i added +1 to.
    Since each query contributes at most +1 to each g it lists (g appears once in a top-k row),
    contribution of query i to g = 1 if (g in tk_raw[i] and g_pid[g]!=q_pid[i]) else 0."""
    H = np.zeros(Ng, dtype=np.int64)
    Nq, k = tk_raw.shape
    for col in range(k):
        gj = tk_raw[:, col]
        sel = g_pid[gj] != q_pid
        np.add.at(H, gj[sel], 1)
    return H


def query_hub_mass_loo(tk_raw, H, q_pid, g_pid):
    """M_loo(q) = sum_{g in topk(q), y_g != y_q} (H[g] - [q contributed to g]).
    Since q contributes exactly +1 to each NEG g in its own top-k, and M sums over exactly
    those NEG g, the self-contribution is exactly 1 per term => M_loo = M_raw - (#neg in topk).
    We implement it directly (per-term subtract 1) to be transparent and robust."""
    Nq, k = tk_raw.shape
    M_raw = np.zeros(Nq, dtype=np.float64)
    M_loo = np.zeros(Nq, dtype=np.float64)
    for col in range(k):
        gj = tk_raw[:, col]
        neg = g_pid[gj] != q_pid
        h = np.where(neg, H[gj], 0.0)
        M_raw += h
        # each such neg g got +1 from THIS query => subtract 1 from its H for the LOO mass
        M_loo += np.where(neg, np.maximum(H[gj] - 1, 0.0), 0.0)
    return M_raw, M_loo


def query_hub_mass_from_external_H(tk_raw, H_ext, q_pid, g_pid):
    """M(q) using an EXTERNALLY estimated H (e.g. from a disjoint query split)."""
    Nq, k = tk_raw.shape
    M = np.zeros(Nq, dtype=np.float64)
    for col in range(k):
        gj = tk_raw[:, col]
        neg = g_pid[gj] != q_pid
        M += np.where(neg, H_ext[gj], 0.0)
    return M


# ============================================================ k-reciprocal (numpy)
def kreciprocal_rerank(qf, gf, k1=20, k2=6, lam=0.3):
    qf = qf.astype(np.float32); gf = gf.astype(np.float32)
    Nq, Ng = qf.shape[0], gf.shape[0]
    allf = np.concatenate([qf, gf], 0)
    orig = np.maximum(2.0 - 2.0 * (allf @ allf.T), 0.0)  # sq-euclidean on unit sphere
    N = Nq + Ng
    initial_rank = np.argsort(orig, axis=1).astype(np.int32)
    V = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        fwd = initial_rank[i, :k1 + 1]
        recip = [c for c in fwd if i in initial_rank[c, :k1 + 1]]
        recip = np.array(recip, dtype=np.int32) if recip else np.array([i], np.int32)
        recip_exp = list(recip)
        for c in recip:
            ch = initial_rank[c, :int(np.around(k1 / 2.0)) + 1]
            if len(np.intersect1d(ch, recip)) > 2.0 / 3.0 * len(ch):
                recip_exp.extend(ch.tolist())
        recip_exp = np.unique(np.array(recip_exp, dtype=np.int32))
        w = np.exp(-orig[i, recip_exp])
        V[i, recip_exp] = (w / w.sum()).astype(np.float32)
    if k2 > 1:
        V = np.array([V[initial_rank[i, :k2]].mean(0) for i in range(N)], dtype=np.float32)
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
    return (1.0 - lam) * jaccard + lam * orig[:Nq, Nq:]


# ============================================================ MAIN
def main():
    t0 = time.time()
    print("#" * 90)
    print(f"# HUBNESS VERIFY P0+P4   dataset={cli.dataset}   k_main={cli.k_main}   feat={cli.cache_feat}")
    print("#" * 90)

    z = np.load(cli.cache_feat, allow_pickle=True)
    qf = z['q_feat'].astype(np.float32); gf = z['g_feat'].astype(np.float32)
    q_pid, q_cam = z['q_pid'].copy(), z['q_cam'].copy()
    g_pid, g_cam = z['g_pid'].copy(), z['g_cam'].copy()
    # drop junk gallery (market distractors pid==-1)
    keep_g = g_pid != -1
    gf, g_pid, g_cam = gf[keep_g], g_pid[keep_g], g_cam[keep_g]
    qf /= (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
    gf /= (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
    Nq, Ng = qf.shape[0], gf.shape[0]
    km = cli.k_main
    print(f"[data] Nq={Nq} Ng={Ng} dim={qf.shape[1]} "
          f"#q-pids={len(np.unique(q_pid))} #g-pids={len(np.unique(g_pid))}")

    sim = qf @ gf.T
    dm = 1.0 - sim
    base = eval_map(dm, q_pid, q_cam, g_pid, g_cam)
    print(f"[sanity] frozen cosine mAP={base['mAP']:.2f} R1={base['r1']:.2f} nq={base['nq']}")

    info = per_query_ap_and_topk(sim, q_pid, q_cam, g_pid, g_cam, km)
    aps = info['aps']; tk_raw = info['tk_raw']
    err = 1.0 - aps
    valid = aps >= 0
    nvalid = int(valid.sum())
    print(f"[valid] queries with >=1 valid positive = {nvalid}/{Nq}")

    # ---- H_k (full) and the ORIGINAL (potentially circular) M ----
    H_full = compute_Hk_neg_with_attrib(tk_raw, q_pid, g_pid, Ng)
    M_raw, M_loo = query_hub_mass_loo(tk_raw, H_full, q_pid, g_pid)

    rng = np.random.RandomState(cli.seed)
    print("\n" + "=" * 90)
    print("P0a  LEAVE-ONE-QUERY-OUT  (subtract q's own +1 from every H_k term in its M)")
    print("=" * 90)
    r_raw, n0 = spearman(err[valid], M_raw[valid])
    r_loo, _ = spearman(err[valid], M_loo[valid])
    p_raw = perm_pvalue(err[valid], M_raw[valid], r_raw, cli.n_perm, rng)
    p_loo = perm_pvalue(err[valid], M_loo[valid], r_loo, cli.n_perm, rng)
    lo_raw, hi_raw = boot_ci_spearman(err[valid], M_raw[valid], cli.n_boot, rng)
    lo_loo, hi_loo = boot_ci_spearman(err[valid], M_loo[valid], cli.n_boot, rng)
    # how big is the self term vs the mass? (the per-query #neg subtracted)
    n_neg_in_topk = np.array([(g_pid[tk_raw[i]] != q_pid[i]).sum() for i in range(Nq)], float)
    frac_self = np.nanmedian((M_raw[valid] - M_loo[valid]) / np.maximum(M_raw[valid], 1e-9))
    print(f"  rho(AP-err, M_raw)  = {r_raw:+.4f}  [95% CI {lo_raw:+.3f},{hi_raw:+.3f}]  perm-p={p_raw:.4f}  (n={n0})")
    print(f"  rho(AP-err, M_loo)  = {r_loo:+.4f}  [95% CI {lo_loo:+.3f},{hi_loo:+.3f}]  perm-p={p_loo:.4f}")
    print(f"  delta rho (raw->loo) = {r_loo - r_raw:+.4f}   "
          f"(median self-term share of M = {100*frac_self:.1f}% of the mass)")
    print(f"  >> P0a: diagnostic {'SURVIVES' if (np.isfinite(r_loo) and r_loo >= 0.8*r_raw and r_loo>0) else 'WEAKENS/COLLAPSES'} "
          f"under LOO de-circularization.")

    print("\n" + "=" * 90)
    print(f"P0b  HELD-OUT SPLIT  (estimate H_k on split A, predict AP-err of split B's M)")
    print("=" * 90)
    # Primary: one seeded 50/50 split. Then average over n_splits for robustness.
    def heldout_rho(seed):
        rs = np.random.RandomState(seed)
        perm = rs.permutation(Nq)
        A = perm[:Nq // 2]; B = perm[Nq // 2:]
        # estimate H_k from split A queries only
        H_A = compute_Hk_neg_with_attrib(tk_raw[A], q_pid[A], g_pid, Ng)
        # M for split B from H_A (B never contributed to H_A => no self-loop)
        M_B = query_hub_mass_from_external_H(tk_raw[B], H_A, q_pid[B], g_pid)
        vB = valid[B]
        r, n = spearman(err[B][vB], M_B[vB])
        return r, n, err[B][vB], M_B[vB]
    r_b, n_b, eB, MB = heldout_rho(cli.seed)
    p_b = perm_pvalue(eB, MB, r_b, cli.n_perm, rng)
    lo_b, hi_b = boot_ci_spearman(eB, MB, cli.n_boot, rng)
    multi = np.array([heldout_rho(cli.seed + 100 + s)[0] for s in range(cli.n_splits)])
    print(f"  primary split (seed={cli.seed}): rho={r_b:+.4f}  [95% CI {lo_b:+.3f},{hi_b:+.3f}]  "
          f"perm-p={p_b:.4f}  (n={n_b})")
    print(f"  over {cli.n_splits} random splits: rho mean={multi.mean():+.4f}  std={multi.std():.4f}  "
          f"min={multi.min():+.4f}  max={multi.max():+.4f}")
    print(f"  >> P0b: held-out (no self-loop possible) rho {'HOLDS' if multi.mean()>0 and np.isfinite(multi.mean()) else 'COLLAPSES'}.")

    print("\n" + "=" * 90)
    print("P0c  STRONGER CHEAP CONTROLS  (partial rho of M_loo | strong retrieval proxies)")
    print("=" * 90)
    proxies = {
        '#false-in-topk': info['n_false_topk'],
        'topk-precision': info['topk_prec'],
        'first-positive-rank': info['first_pos_rank'],
        'mean-negative-sim': info['mean_neg_sim'],
        'top1-correct': info['top1_correct'],
        'same-cam-frac': info['same_cam_frac'],
    }
    # individual marginal correlations of err with each proxy (context)
    print("  marginal rho(AP-err, proxy):")
    for nm, v in proxies.items():
        r, _ = spearman(err[valid], v[valid])
        print(f"    {nm:<22} {r:+.4f}")
    print("  --- partial rho(AP-err, M_loo | proxy) [does hub mass add signal beyond it?] ---")
    for nm, v in proxies.items():
        pr, _ = partial_spearman(err[valid], M_loo[valid], v[valid])
        print(f"    M_loo | {nm:<20} = {pr:+.4f}")
    # joint: control ALL strong proxies at once
    cov = np.column_stack([proxies[k][valid] for k in proxies])
    pr_all, n_all = partial_spearman(err[valid], M_loo[valid], cov)
    print(f"  M_loo | ALL {len(proxies)} strong proxies = {pr_all:+.4f}  (n={n_all})")
    print(f"  >> P0c: M_loo {'ADDS independent signal' if (np.isfinite(pr_all) and abs(pr_all)>=0.05) else 'is REDUCIBLE'} "
          f"beyond strong retrieval proxies.")

    # ============================================================ P4
    print("\n" + "=" * 90)
    print("P4  k-RECIPROCAL REFRAME  (is M(q) what re-ranking repairs?)")
    print("=" * 90)
    print(f"  computing k-reciprocal (k1={cli.rr_k1}, k2={cli.rr_k2}, lam={cli.rr_lam}) ...", flush=True)
    rr = kreciprocal_rerank(qf, gf, k1=cli.rr_k1, k2=cli.rr_k2, lam=cli.rr_lam)
    r_rr = eval_map(rr, q_pid, q_cam, g_pid, g_cam)
    aps_base = per_query_ap_from_dist(dm, q_pid, q_cam, g_pid, g_cam)
    aps_rr = per_query_ap_from_dist(rr, q_pid, q_cam, g_pid, g_cam)
    gain = aps_rr - aps_base
    sel = (aps_base >= 0) & (aps_rr >= 0)
    print(f"  k-reciprocal mAP={r_rr['mAP']:.3f} (base {base['mAP']:.3f}, d{r_rr['mAP']-base['mAP']:+.3f})")
    r_gM, _ = spearman(M_raw[sel], gain[sel])
    r_gMloo, _ = spearman(M_loo[sel], gain[sel])
    p_gM = perm_pvalue(M_raw[sel], gain[sel], r_gM, cli.n_perm, rng)
    lo_g, hi_g = boot_ci_spearman(M_raw[sel], gain[sel], cli.n_boot, rng)
    print(f"  rho(M_raw, k-recip per-query gain) = {r_gM:+.4f}  [95% CI {lo_g:+.3f},{hi_g:+.3f}]  perm-p={p_gM:.4f}")
    print(f"  rho(M_loo, k-recip per-query gain) = {r_gMloo:+.4f}")
    # binned trend by M quartile
    order = np.argsort(M_raw[sel]); qb = np.array_split(order, 5)
    Msel, gsel = M_raw[sel], gain[sel]
    print("  M(q) quintile -> mean k-reciprocal AP gain:")
    for b, idxs in enumerate(qb):
        print(f"    Q{b} (n={len(idxs):4d}, mean M={Msel[idxs].mean():10.1f}): "
              f"mean gain = {100*gsel[idxs].mean():+.3f}  (frac improved {100*(gsel[idxs]>0).mean():.1f}%)")
    print(f"  >> P4: high-M queries are {'PRECISELY the ones k-reciprocal repairs most (reframe holds)' if r_gM>0.1 else 'NOT specially repaired by k-reciprocal'}.")

    print(f"\n[done] {cli.dataset} in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
