#!/usr/bin/env python3
"""Evidence-Sufficiency / Weak-Positive-Support re-frame — ZERO-TRAINING kill-switch.

UNIFYING RE-FRAME UNDER TEST
----------------------------
ReID failure (incl. the gallery-growth tax) is driven by the QUERY's WEAK POSITIVE
SUPPORT — too little visible, cross-camera-reproducible identity evidence — NOT by a
strong distractor. Re-definition: don't ask "how to suppress the distractor", ask
"does this query have enough visible evidence to be reliably retrieved".

This connects two things:
  * the just-killed Gallery-Growth Tax remedy (its precheck showed OD's tax is
    WEAK-POSITIVE driven, not distractor driven), and
  * the d14 "Evidence-Sufficient ReID" backup (single-image support insufficiency,
    rescued by oracle multi-query evidence union).

THREE tests, ALL on FROZEN features (no backward, numpy only):

  TEST 1 (positive-support explains the TAX residual):
      Reuse the precheck's core/pool split + disjoint label/visible distractor halves.
      Compute per-query POSITIVE-SUPPORT quantities from the 1x core gallery only:
        - lower-tail positive sim  (soft-min over same-ID positives) -> LOW = weak
        - positive dispersion      (std of positive sims)            -> HIGH = scattered
        - #cross-camera positives  (count)                           -> LOW = weak
      Does positive-support explain the gallery-growth tax residual (the 1x->10x AP
      drop) AFTER controlling 1x-top1-margin AND #false-in-topk?
      KEY (HUBNESS §7.6 lesson): two queries with similar 1x margin but different
      positive-support must show different tax (partial Spearman survives controls).

  TEST 2 (positive-support predicts per-query FAILURE):
      ROC-AUC of positive-support predicting per-query AP-failure on the FULL gallery,
      vs trivial (1x-top1-margin / #false-in-topk / feat-norm). Must clearly beat
      trivial in raw AUC AND in INCREMENTAL OOF-AUC after controlling them.

  TEST 3 (oracle MULTI-QUERY recovery — the d14 core):
      For LOW positive-support FAILURE queries, add a 2nd same-ID image and form an
      evidence UNION (mean / max-pool of the two query feats). Does AP recover?
      CONTROLS:
        - add a RANDOM cross-ID image instead (must NOT recover -> the gain is
          identity evidence, not just "two probes average out noise");
        - add 2nd same-ID image vs k-RECIPROCAL re-rank on the single query (is the
          recovery a unique multi-query effect or does free re-rank already do it?).
      If same-ID union recovers >> random-ID AND is not fully covered by k-reciprocal
      -> the failure is EVIDENCE INSUFFICIENCY (more evidence fixes it), supporting d14.

VERDICT: positive-support has an INDEPENDENT-of-trivial signal (survives TEST1 partial
& TEST2 incremental) AND oracle multi-query proves "evidence insufficiency" (TEST3).

Reuses the kill-switch feature caches + eval/stat helpers + core/pool split.
Run on lab-3090-d (cached frozen features, pure numpy, no GPU training):
  cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 python3 \
    experiments/cargo_cvpb/cvpb_evidence_killswitch.py \
    --dataset market1501 --cache_feat /tmp/hub_market_feats.npz \
    2>&1 | tee /tmp/cvpb_evidence_market.log
  # OD: --dataset occluded_duke --cache_feat /tmp/hub_oduke_feats.npz
"""
import os, sys, time, argparse
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.abspath(os.path.join(_here, '..', '..'))
sys.path.insert(0, _repo)

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', default='market1501')
ap.add_argument('--cache_feat', default='/tmp/hub_market_feats.npz')
ap.add_argument('--seed', type=int, default=42)
# --- TEST 1 core/pool split (identical convention to precheck / kill-switch test_A) ---
ap.add_argument('--core_frac', type=float, default=0.2,
                help='fraction of query-IDs kept as CORE task; rest (+gallery-only IDs) = distractor pool')
ap.add_argument('--core_cap', type=int, default=8,
                help='cap core gallery imgs/ID so the held-out pool can reach 10x (positives kept)')
ap.add_argument('--max_mult', type=float, default=10.0, help='target gallery multiplier for the tax label')
ap.add_argument('--n_seeds', type=int, default=5, help='resample distractor draws for the 10x AP, average')
# soft-min temperature for lower-tail positive sim
ap.add_argument('--a_temp', type=float, default=20.0)
ap.add_argument('--topk', type=int, default=10, help='k for #false-in-topk trivial proxy')
# --- TEST 2 failure label on the FULL gallery ---
ap.add_argument('--fail_quant', type=float, default=0.30, help='bottom-q AP = failure (full-gallery)')
# --- TEST 3 oracle multi-query ---
ap.add_argument('--low_support_quant', type=float, default=0.30,
                help='bottom-q by positive-support among FAILURE queries -> the rescue target set')
ap.add_argument('--krecip_k1', type=int, default=20)
ap.add_argument('--krecip_k2', type=int, default=6)
ap.add_argument('--krecip_lambda', type=float, default=0.3)
cli = ap.parse_args()
np.random.seed(cli.seed)
RNG = np.random.RandomState(cli.seed)


# =========================================================================== #
# DATA LOAD (reuse kill-switch cache + normalization)
# =========================================================================== #
def load_data():
    z = np.load(cli.cache_feat, allow_pickle=True)
    q = dict(feat=z['q_feat'], pid=z['q_pid'], cam=z['q_cam'], name=z['q_name'])
    g = dict(feat=z['g_feat'], pid=z['g_pid'], cam=z['g_cam'], name=z['g_name'])
    keep = g['pid'] != -1                              # drop market junk distractors
    for k in ('feat', 'pid', 'cam', 'name'):
        g[k] = g[k][keep]
    q['feat'] = q['feat'].astype(np.float32)
    g['feat'] = g['feat'].astype(np.float32)
    q['feat'] /= (np.linalg.norm(q['feat'], axis=1, keepdims=True) + 1e-12)
    g['feat'] /= (np.linalg.norm(g['feat'], axis=1, keepdims=True) + 1e-12)
    print(f"[data] {cli.cache_feat}: Nq={len(q['name'])} Ng={len(g['name'])} dim={q['feat'].shape[1]} "
          f"#q-IDs={len(np.unique(q['pid']))} #g-IDs={len(np.unique(g['pid']))}", flush=True)
    return q, g


# =========================================================================== #
# per-query AP (+ optional #false-in-topk). Market protocol: drop same pid&cam junk.
# Works on gallery SUBSETS. dist = 1 - cosine.
# =========================================================================== #
def per_query_ap(qf, gf, q_pid, q_cam, g_pid, g_cam, topk=None, return_false=False):
    sim = qf @ gf.T
    dm = 1.0 - sim
    nq = dm.shape[0]
    order_all = np.argsort(dm, axis=1)
    aps = np.full(nq, -1.0)
    false_k = np.full(nq, -1.0)
    k = topk if topk is not None else cli.topk
    for i in range(nq):
        order = order_all[i]
        keep = ~((g_pid[order] == q_pid[i]) & (g_cam[order] == q_cam[i]))
        gp = g_pid[order][keep]
        m = (gp == q_pid[i]).astype(np.int32)
        if not m.any():
            continue
        prec = m.cumsum() / (np.arange(len(m)) + 1.0)
        aps[i] = (prec * m).sum() / m.sum()
        if return_false:
            false_k[i] = int((gp[:k] != q_pid[i]).sum())
    if return_false:
        return aps, false_k
    return aps


# =========================================================================== #
# STAT helpers (tie-aware Spearman / partial Spearman / ROC-AUC / OOF logistic)
# Copied from the precheck so TEST1/2 use IDENTICAL machinery to the Tax precheck.
# =========================================================================== #
def _tied_rank(v):
    v = np.asarray(v, float)
    order = np.argsort(v, kind='mergesort')
    ranks = np.empty(len(v), float)
    sv = v[order]
    i, n = 0, len(v)
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0   # midrank (1-based)
        i = j + 1
    return ranks


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float('nan'), 0
    rx, ry = _tied_rank(x), _tied_rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx**2).sum() * (ry**2).sum())
    return (float((rx * ry).sum() / den) if den > 0 else float('nan')), len(x)


def partial_spearman(x, y, Z):
    """Spearman(x, y | Z) — rank-partial correlation controlling columns of Z."""
    x = np.asarray(x, float); y = np.asarray(y, float); Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(Z).all(axis=1)
    x, y, Z = x[ok], y[ok], Z[ok]
    if len(x) < 5:
        return float('nan'), 0
    rx, ry = _tied_rank(x), _tied_rank(y)
    Zr = np.column_stack([np.ones(len(x))] + [_tied_rank(Z[:, j]) for j in range(Z.shape[1])])
    resid = lambda r: r - Zr @ np.linalg.lstsq(Zr, r, rcond=None)[0]
    ex, ey = resid(rx), resid(ry)
    den = np.sqrt((ex**2).sum() * (ey**2).sum())
    return (float((ex * ey).sum() / den) if den > 0 else float('nan')), len(x)


def roc_auc(risk, label):
    """AUC that risk (higher = more likely positive) ranks positives above negatives."""
    risk = np.asarray(risk, float); label = np.asarray(label, bool)
    ok = np.isfinite(risk)
    risk, label = risk[ok], label[ok]
    n_pos = int(label.sum()); n_neg = int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan'), n_pos, n_neg
    r = _tied_rank(risk)
    auc = (r[label].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc), n_pos, n_neg


def auc_ci_boot(risk, label, nboot=800, seed=0):
    risk = np.asarray(risk, float); label = np.asarray(label, bool)
    ok = np.isfinite(risk); risk, label = risk[ok], label[ok]
    rs = np.random.RandomState(seed); n = len(risk); vals = []
    for _ in range(nboot):
        idx = rs.randint(0, n, n)
        if label[idx].sum() == 0 or (~label[idx]).sum() == 0:
            continue
        a, _, _ = roc_auc(risk[idx], label[idx]); vals.append(a)
    if not vals:
        return float('nan'), float('nan')
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _standardize(X):
    X = np.asarray(X, float)
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-9] = 1.0
    return (X - mu) / sd, mu, sd


def logreg_fit(X, y, l2=1.0, iters=200):
    Xs, mu, sd = _standardize(X)
    n, d = Xs.shape
    Z = np.column_stack([np.ones(n), Xs])
    w = np.zeros(d + 1)
    R = np.eye(d + 1) * l2; R[0, 0] = 0.0
    for _ in range(iters):
        eta = Z @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        Wd = np.clip(p * (1 - p), 1e-6, None)
        grad = Z.T @ (p - y) + R @ w
        H = Z.T @ (Z * Wd[:, None]) + R
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        w_new = w - step
        if np.max(np.abs(w_new - w)) < 1e-7:
            w = w_new; break
        w = w_new
    return w, mu, sd


def logreg_score(X, w, mu, sd):
    Xs = (np.asarray(X, float) - mu) / sd
    Z = np.column_stack([np.ones(len(Xs)), Xs])
    eta = Z @ w
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))


def cv_auc_logreg(X, y, nfold=5, l2=1.0, seed=0):
    X = np.asarray(X, float); y = np.asarray(y, float)
    n = len(y)
    rs = np.random.RandomState(seed)
    perm = rs.permutation(n)
    folds = np.array_split(perm, nfold)
    oof = np.full(n, np.nan)
    for k in range(nfold):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(nfold) if j != k])
        if y[tr].sum() == 0 or (1 - y[tr]).sum() == 0:
            continue
        w, mu, sd = logreg_fit(X[tr], y[tr], l2=l2)
        oof[te] = logreg_score(X[te], w, mu, sd)
    a, _, _ = roc_auc(oof, y.astype(bool))
    return a


def softmin(s, a):
    """soft-min = -1/a * logsumexp(-a*s). Smooth minimum of s (lower-tail value)."""
    s = np.asarray(s, float)
    if len(s) == 0:
        return np.nan
    m = (-a * s).max()
    return -(m + np.log(np.exp(-a * s - m).sum())) / a


# =========================================================================== #
# POSITIVE-SUPPORT quantities (the re-frame's variables) — computed from a given
# (query, positive-gallery) similarity context. Used both at 1x (TEST1) and full
# gallery (TEST2). Returns dict of per-query arrays.
# =========================================================================== #
def positive_support(qf, q_pid, q_cam, gf, g_pid, g_cam, a_temp):
    """For each query: lower-tail positive sim (soft-min), positive dispersion (std),
    #cross-camera positives, mean positive sim. Junk-removed (same pid & cam dropped)."""
    nq = len(q_pid)
    sim = qf @ gf.T                                     # nq x Ng (positives sparse)
    lowtail = np.full(nq, np.nan)
    disp = np.full(nq, np.nan)
    ncc = np.full(nq, np.nan)
    meanpos = np.full(nq, np.nan)
    for i in range(nq):
        keep_pos = (g_pid == q_pid[i]) & (g_cam != q_cam[i])   # cross-camera positives only
        s_pos = sim[i][keep_pos]
        npos = len(s_pos)
        ncc[i] = npos
        if npos == 0:
            continue
        lowtail[i] = softmin(s_pos, a_temp)             # weakest cross-cam positive (smooth)
        meanpos[i] = float(s_pos.mean())
        disp[i] = float(s_pos.std()) if npos >= 2 else 0.0
    return dict(lowtail=lowtail, disp=disp, ncc=ncc, meanpos=meanpos)


# =========================================================================== #
# core/pool split (IDENTICAL to precheck) — returns the pieces TEST1 needs
# =========================================================================== #
def build_core_pool(q, g):
    qf, q_pid, q_cam = q['feat'], q['pid'], q['cam']
    gf, g_pid, g_cam = g['feat'], g['pid'], g['cam']
    q_ids = np.unique(q_pid); g_ids = np.unique(g_pid)
    gallery_only_ids = set(g_ids.tolist()) - set(q_ids.tolist())
    RNG.shuffle(q_ids)
    n_core = max(1, int(round(cli.core_frac * len(q_ids))))
    core_ids = set(q_ids[:n_core].tolist())
    pool_ids = set(q_ids[n_core:].tolist()) | gallery_only_ids
    qsel = np.array([p in core_ids for p in q_pid])
    cqf, cq_pid, cq_cam = qf[qsel], q_pid[qsel], q_cam[qsel]
    core_idx_list = []
    for cid in core_ids:
        idx = np.where(g_pid == cid)[0]
        if len(idx) > cli.core_cap:
            cams_here = g_cam[idx]
            chosen = []
            for c in np.unique(cams_here):
                cidx = idx[cams_here == c]
                chosen.append(cidx[RNG.randint(len(cidx))])
                if len(chosen) >= cli.core_cap:
                    break
            chosen = np.array(chosen)
            if len(chosen) < cli.core_cap:
                rest = np.setdiff1d(idx, chosen)
                extra = RNG.choice(rest, min(cli.core_cap - len(chosen), len(rest)), replace=False)
                chosen = np.concatenate([chosen, extra])
            idx = chosen
        core_idx_list.append(idx)
    core_idx = np.concatenate(core_idx_list)
    pool_idx_all = np.where(np.array([p in pool_ids for p in g_pid]))[0]
    return dict(qsel=qsel, cqf=cqf, cq_pid=cq_pid, cq_cam=cq_cam,
                core_idx=core_idx, pool_idx_all=pool_idx_all,
                core_ids=core_ids, pool_ids=pool_ids,
                n_gallery_only=len(gallery_only_ids))


# =========================================================================== #
# TEST 1 — POSITIVE-SUPPORT explains the GALLERY-GROWTH TAX residual
# =========================================================================== #
def test_1(q, g):
    print("\n" + "#" * 80)
    print("# TEST 1 — POSITIVE-SUPPORT vs GALLERY-GROWTH-TAX residual "
          "(control 1x-top1-margin + #false-in-topk)")
    print("#" * 80)
    gf, g_pid, g_cam = g['feat'], g['pid'], g['cam']
    cp = build_core_pool(q, g)
    cqf, cq_pid, cq_cam = cp['cqf'], cp['cq_pid'], cp['cq_cam']
    core_idx, pool_idx_all = cp['core_idx'], cp['pool_idx_all']
    Ng_core, Npool = len(core_idx), len(pool_idx_all)
    n_add_target = int(round((cli.max_mult - 1.0) * Ng_core))
    max_mult_real = 1.0 + Npool / Ng_core
    print(f"[1] core-IDs={len(cp['core_ids'])} pool-IDs={len(cp['pool_ids'])} "
          f"(incl {cp['n_gallery_only']} gallery-only)")
    print(f"[1] core queries={len(cq_pid)} core gallery={Ng_core} (cap {cli.core_cap}/ID) "
          f"pool imgs={Npool}  max-achievable={max_mult_real:.2f}x")

    # disjoint label/visible distractor halves (precheck convention) — so the tax label
    # and the trivial #false / margin features never share the exact same distractors.
    pool_perm = RNG.permutation(pool_idx_all)
    half = len(pool_perm) // 2
    pool_label = pool_perm[:half]
    pool_visible = pool_perm[half:]
    print(f"[1] distractor pool DISJOINT: label-half={len(pool_label)} visible-half={len(pool_visible)}")

    # 1x baseline AP (+ #false-in-topk) per core query
    base_aps, base_false = per_query_ap(cqf, gf[core_idx], cq_pid, cq_cam,
                                        g_pid[core_idx], g_cam[core_idx],
                                        topk=cli.topk, return_false=True)
    valid = base_aps >= 0
    print(f"[1] 1x baseline (core only): mAP={base_aps[valid].mean()*100:.3f} nq={int(valid.sum())}")

    # 10x AP per core query (avg over draws from LABEL half) -> the tax
    n_add_lbl = min(n_add_target, len(pool_label))
    big_mult = 1.0 + n_add_lbl / Ng_core
    runs = []
    for s in range(cli.n_seeds):
        rs = np.random.RandomState(cli.seed + 1000 * s + 7)
        add = pool_label if n_add_lbl >= len(pool_label) else rs.choice(pool_label, n_add_lbl, replace=False)
        gidx = np.concatenate([core_idx, add])
        runs.append(per_query_ap(cqf, gf[gidx], cq_pid, cq_cam, g_pid[gidx], g_cam[gidx]))
    aps_10x = np.mean(np.array(runs), axis=0)
    d_ap = (aps_10x - base_aps) * 100.0                 # negative = dropped
    tax = -d_ap                                         # tax residual: HIGH = bigger drop
    print(f"[1] {big_mult:.1f}x (label half, {cli.n_seeds} draws): mAP={aps_10x[valid].mean()*100:.3f} "
          f"dmAP={(aps_10x[valid].mean()-base_aps[valid].mean())*100:+.3f}  (tax = -dAP)")

    # POSITIVE-SUPPORT (from 1x core gallery: cross-cam positives only)
    ps = positive_support(cqf, cq_pid, cq_cam, gf[core_idx], g_pid[core_idx], g_cam[core_idx], cli.a_temp)
    # weak-positive RISK convention: higher = weaker support = predict bigger tax
    risk_lowtail = -ps['lowtail']        # low lower-tail sim -> weak
    risk_disp = ps['disp']               # high dispersion -> scattered/weak
    risk_ncc = -ps['ncc']                # few cross-cam positives -> weak

    # TRIVIAL controls (deploy-visible, 1x): top1-margin vs visible distractor pool + #false-in-topk
    sim_core = cqf @ gf[core_idx].T
    sim_vis = cqf @ gf[pool_visible].T
    cg_pid, cg_cam = g_pid[core_idx], g_cam[core_idx]
    triv_top1margin = np.full(len(cq_pid), np.nan)
    for i in range(len(cq_pid)):
        keep_pos = (cg_pid == cq_pid[i]) & (cg_cam != cq_cam[i])
        s_pos = sim_core[i][keep_pos]
        if len(s_pos) == 0:
            continue
        triv_top1margin[i] = s_pos.max() - sim_vis[i].max()   # best pos - best visible distractor
    risk_margin = -triv_top1margin       # small margin -> fragile
    risk_false = base_false              # more false-in-topk at 1x -> harder

    ev = valid
    print(f"\n[1] raw Spearman(positive-support risk, tax) over {int(ev.sum())} valid core queries:")
    for nm, r in [('lowtail-pos(soft-min)', risk_lowtail), ('pos-dispersion', risk_disp),
                  ('#cross-cam-pos(neg)', risk_ncc),
                  ('[triv]1x-top1-margin', risk_margin), ('[triv]#false-in-topk', risk_false)]:
        rho, n = spearman(r[ev], tax[ev])
        print(f"     {nm:>24}: rho={rho:+.4f} (n={n})")

    # ★ LIFE-OR-DEATH partials: positive-support vs tax controlling 1x-margin AND #false
    Zc = np.column_stack([risk_margin[ev], risk_false[ev]])
    print(f"\n[1] ★PARTIAL Spearman(support-risk, tax | 1x-top1-margin + #false-in-topk):")
    for nm, r in [('lowtail-pos(soft-min)', risk_lowtail), ('pos-dispersion', risk_disp),
                  ('#cross-cam-pos(neg)', risk_ncc)]:
        pr, n = partial_spearman(r[ev], tax[ev], Zc)
        raw, _ = spearman(r[ev], tax[ev])
        print(f"     {nm:>24}: partial={pr:+.4f}  (raw={raw:+.4f}, n={n})  "
              f"{'<< survives' if abs(pr) > 0.10 and pr*raw>0 else '<< collapsed' }")
    # reverse direction (do the trivials survive controlling support? — fairness check)
    Zs = np.column_stack([risk_lowtail[ev], risk_ncc[ev]])
    pr_m, _ = partial_spearman(risk_margin[ev], tax[ev], Zs)
    pr_f, _ = partial_spearman(risk_false[ev], tax[ev], Zs)
    print(f"     [reverse] 1x-margin | support  = {pr_m:+.4f}   #false | support = {pr_f:+.4f}")

    # combined: does support add to a logistic predicting big-tax over trivials? (OOF AUC)
    nflag = int(round(0.30 * int(ev.sum())))
    vidx = np.where(ev)[0]
    jit = np.random.RandomState(cli.seed + 9).rand(len(vidx)) * 1e-9
    order = np.argsort(-tax[vidx] + jit)               # NOT failure-first: pick biggest tax
    big_tax = np.zeros_like(valid)
    big_tax[vidx[order[:nflag]]] = True
    y = big_tax[ev].astype(float)
    Xtriv = np.column_stack([risk_margin[ev], risk_false[ev]])
    Xsupp = np.column_stack([risk_lowtail[ev], risk_disp[ev], risk_ncc[ev]])
    ok = np.isfinite(Xtriv).all(1) & np.isfinite(Xsupp).all(1)
    a_triv = cv_auc_logreg(Xtriv[ok], y[ok], seed=cli.seed)
    a_both = cv_auc_logreg(np.column_stack([Xtriv[ok], Xsupp[ok]]), y[ok], seed=cli.seed)
    a_supp = cv_auc_logreg(Xsupp[ok], y[ok], seed=cli.seed)
    print(f"\n[1] big-tax (top-30% tax) OOF-AUC: trivials={a_triv:.4f}  +support={a_both:.4f}  "
          f"support-solo={a_supp:.4f}  >> INCREMENT={a_both-a_triv:+.4f}")
    return dict(partial_lowtail=partial_spearman(risk_lowtail[ev], tax[ev], Zc)[0],
                partial_ncc=partial_spearman(risk_ncc[ev], tax[ev], Zc)[0],
                partial_disp=partial_spearman(risk_disp[ev], tax[ev], Zc)[0],
                incr=a_both - a_triv)


# =========================================================================== #
# TEST 2 — POSITIVE-SUPPORT predicts per-query FAILURE on the FULL gallery
# =========================================================================== #
def test_2(q, g):
    print("\n" + "#" * 80)
    print("# TEST 2 — POSITIVE-SUPPORT predicts per-query FAILURE (full gallery) vs trivial")
    print("#" * 80)
    qf, q_pid, q_cam = q['feat'], q['pid'], q['cam']
    gf, g_pid, g_cam = g['feat'], g['pid'], g['cam']
    aps, false_k = per_query_ap(qf, gf, q_pid, q_cam, g_pid, g_cam,
                                topk=cli.topk, return_false=True)
    valid = aps >= 0
    ev = valid
    # FAILURE label: bottom-q AP among valid queries (rank-based, ties jittered)
    vidx = np.where(ev)[0]
    nflag = int(round(cli.fail_quant * len(vidx)))
    jit = np.random.RandomState(cli.seed + 3).rand(len(vidx)) * 1e-9
    order = np.argsort(aps[vidx] + jit)                # lowest AP first
    fail = np.zeros_like(valid)
    fail[vidx[order[:nflag]]] = True
    print(f"[2] full-gallery mAP={aps[ev].mean()*100:.3f}  failure=bottom-{cli.fail_quant:.0%} AP "
          f"n={int(fail.sum())}/{int(ev.sum())}")

    # positive-support on the FULL gallery (cross-cam positives only)
    ps = positive_support(qf, q_pid, q_cam, gf, g_pid, g_cam, cli.a_temp)
    # NOTE: cached features are L2-normalized at extraction (F.normalize in extract_features),
    # so a raw feat-norm proxy is a degenerate constant here and is intentionally NOT used.
    # The two LIFE-OR-DEATH trivials (HUBNESS §7.6) are 1x-top1-margin and #false-in-topk.
    feats = {
        'lowtail-pos(soft-min)': -ps['lowtail'],       # ★ support: weak = high risk
        'pos-dispersion':         ps['disp'],          # ★ support
        '#cross-cam-pos(neg)':   -ps['ncc'],           # ★ support
        '[triv]1x-top1-margin':   None,                # filled below (needs best-distractor)
        '[triv]#false-in-topk':   false_k,             # trivial
    }
    # full-gallery top1-margin: (best positive sim) - (best distractor sim)
    sim = qf @ gf.T
    top1margin = np.full(len(q_pid), np.nan)
    for i in range(len(q_pid)):
        keep_pos = (g_pid == q_pid[i]) & (g_cam != q_cam[i])
        s_pos = sim[i][keep_pos]
        if len(s_pos) == 0:
            continue
        neg = sim[i][~((g_pid == q_pid[i]))]           # cross-ID sims
        top1margin[i] = s_pos.max() - (neg.max() if len(neg) else 0.0)
    feats['[triv]1x-top1-margin'] = -top1margin

    y = fail[ev].astype(bool)
    print(f"\n  {'predictor':>24}  {'AUC':>7}  {'95% CI':>16}")
    aucs = {}
    for nm, risk in feats.items():
        a, npos, nneg = roc_auc(risk[ev], y)
        lo, hi = auc_ci_boot(risk[ev], y, nboot=800, seed=cli.seed)
        aucs[nm] = a
        star = ' <<SUPPORT' if not nm.startswith('[triv]') else ''
        print(f"  {nm:>24}  {a:>7.4f}  [{lo:>6.3f},{hi:>6.3f}]{star}")

    # vs-trivial incremental (OOF logistic) — HUBNESS §7.6 discipline
    triv_names = ['[triv]1x-top1-margin', '[triv]#false-in-topk']
    supp_names = ['lowtail-pos(soft-min)', 'pos-dispersion', '#cross-cam-pos(neg)']
    Xtriv = np.column_stack([feats[n][ev] for n in triv_names])
    Xsupp = np.column_stack([feats[n][ev] for n in supp_names])
    ok = np.isfinite(Xtriv).all(1) & np.isfinite(Xsupp).all(1)
    yk = y[ok].astype(float)
    a_triv = cv_auc_logreg(Xtriv[ok], yk, seed=cli.seed)
    a_both = cv_auc_logreg(np.column_stack([Xtriv[ok], Xsupp[ok]]), yk, seed=cli.seed)
    a_supp = cv_auc_logreg(Xsupp[ok], yk, seed=cli.seed)
    best_supp = max(aucs[n] for n in supp_names)
    best_triv = max(aucs[n] for n in triv_names)
    print(f"\n  -- vs-TRIVIAL incremental (5-fold OOF logistic, HUBNESS §7.6) --")
    print(f"     trivials-only (3 proxies)   OOF-AUC = {a_triv:.4f}")
    print(f"     trivials + support          OOF-AUC = {a_both:.4f}")
    print(f"     support-only (3 proxies)    OOF-AUC = {a_supp:.4f}")
    print(f"     >> INCREMENT support adds on top of trivials = {a_both-a_triv:+.4f}")
    print(f"     >> best support AUC - best trivial AUC        = {best_supp-best_triv:+.4f}")
    # partial spearman: best support var vs continuous (-AP) controlling all trivials
    best_supp_name = max(supp_names, key=lambda n: aucs[n])
    pr, _ = partial_spearman(feats[best_supp_name][ev], (-aps)[ev], Xtriv)
    raw, _ = spearman(feats[best_supp_name][ev], (-aps)[ev])
    print(f"     partial Spearman({best_supp_name}, -AP | 3 trivials) = {pr:+.4f} (raw={raw:+.4f})")
    return dict(aucs=aucs, incr=a_both - a_triv, best_supp_minus_triv=best_supp - best_triv,
                supp_solo=a_supp, partial=pr)


# =========================================================================== #
# TEST 3 — ORACLE MULTI-QUERY recovery (d14 core)
# =========================================================================== #
def kreciprocal_rerank(qf, gf, k1=20, k2=6, lam=0.3):
    """Standard k-reciprocal re-ranking (Zhong et al. CVPR2017), VERBATIM from the
    VALIDATED implementation in hub_verify_p0_p4.py (it reproduced the known +10.98 OD
    / +1.26 Market global gains). Combined query+gallery graph; returns Nq x Ng final
    distance. We run it on the FULL query set then index out the rows we need so each
    query's re-rank uses the real deployment graph context (matching P4)."""
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


def per_query_ap_from_dist(distmat, sel_rows, q_pid, q_cam, g_pid, g_cam):
    """Per-query AP from a precomputed Nq x Ng distance matrix, for the selected query
    rows only (junk removed: same pid & cam dropped)."""
    aps = np.full(len(sel_rows), -1.0)
    for r, i in enumerate(sel_rows):
        order = np.argsort(distmat[i])
        keep = ~((g_pid[order] == q_pid[i]) & (g_cam[order] == q_cam[i]))
        gp = g_pid[order][keep]
        m = (gp == q_pid[i]).astype(np.int32)
        if not m.any():
            continue
        prec = m.cumsum() / (np.arange(len(m)) + 1.0)
        aps[r] = (prec * m).sum() / m.sum()
    return aps


def test_3(q, g):
    print("\n" + "#" * 80)
    print("# TEST 3 — ORACLE MULTI-QUERY recovery on LOW-SUPPORT failures (d14 core)")
    print("#" * 80)
    qf, q_pid, q_cam = q['feat'], q['pid'], q['cam']
    gf, g_pid, g_cam = g['feat'], g['pid'], g['cam']

    # full-gallery per-query AP + failure + positive-support
    aps, _ = per_query_ap(qf, gf, q_pid, q_cam, g_pid, g_cam, return_false=True)
    valid = aps >= 0
    ps = positive_support(qf, q_pid, q_cam, gf, g_pid, g_cam, cli.a_temp)
    support = ps['lowtail']                             # higher = stronger support

    # FAILURE = bottom-30% AP; among failures, LOW-SUPPORT = bottom-q by support
    vidx = np.where(valid)[0]
    nfail = int(round(0.30 * len(vidx)))
    jit = np.random.RandomState(cli.seed + 3).rand(len(vidx)) * 1e-9
    fail_order = np.argsort(aps[vidx] + jit)
    fail_idx = vidx[fail_order[:nfail]]
    # low-support subset among failures (need >=2 same-ID query imgs to do union -> see below)
    supp_fail = support[fail_idx]
    ok_supp = np.isfinite(supp_fail)
    fidx2 = fail_idx[ok_supp]; supp2 = supp_fail[ok_supp]
    nlow = int(round(cli.low_support_quant * len(fidx2)))
    low_order = np.argsort(supp2)                       # weakest support first
    low_support_fail = fidx2[low_order[:nlow]]
    print(f"[3] full mAP={aps[valid].mean()*100:.3f}  failures(bot-30%)={len(fail_idx)}  "
          f"low-support failures(bot-{cli.low_support_quant:.0%})={len(low_support_fail)}")

    # for each low-support failure query, we need a SECOND same-ID query image to union.
    # build map pid -> all query indices.
    pid2q = {}
    for j, p in enumerate(q_pid):
        pid2q.setdefault(int(p), []).append(j)

    def add_second_same_id(qi, rs):
        cands = [j for j in pid2q[int(q_pid[qi])] if j != qi]
        if not cands:
            return None
        return cands[rs.randint(len(cands))]

    def add_random_cross_id(qi, rs):
        while True:
            j = rs.randint(len(q_pid))
            if int(q_pid[j]) != int(q_pid[qi]):
                return j

    # ORACLE union: mean of the two L2-normalized query feats (renormalized). Also report max-pool.
    # base AP and same-ID/random union are all computed against the SAME full gallery so the only
    # thing that changes is the query representation (clean evidence-amount manipulation).
    # The random-ID control is averaged over N_RAND draws so a single lucky/unlucky cross-ID
    # partner can't fake (or mask) a "recovery".
    N_RAND = 5
    rs = np.random.RandomState(cli.seed + 17)
    base_list, union_mean_list, union_max_list, rand_union_list = [], [], [], []
    used = []
    for qi in low_support_fail:
        j2 = add_second_same_id(qi, rs)
        if j2 is None:
            continue                                    # singleton query ID -> can't union
        used.append(qi)
        # single-query AP (recompute for this row)
        ap1 = per_query_ap(qf[qi:qi+1], gf, q_pid[qi:qi+1], q_cam[qi:qi+1], g_pid, g_cam)[0]
        # same-ID evidence union (mean / max)
        f_mean = qf[qi] + qf[j2]; f_mean /= (np.linalg.norm(f_mean) + 1e-12)
        f_max = np.maximum(qf[qi], qf[j2]); f_max /= (np.linalg.norm(f_max) + 1e-12)
        ap_mean = per_query_ap(f_mean[None], gf, q_pid[qi:qi+1], q_cam[qi:qi+1], g_pid, g_cam)[0]
        ap_max = per_query_ap(f_max[None], gf, q_pid[qi:qi+1], q_cam[qi:qi+1], g_pid, g_cam)[0]
        # CONTROL: union with a RANDOM cross-ID query image (must NOT recover), avg over draws.
        ap_rand_draws = []
        for _ in range(N_RAND):
            jr = add_random_cross_id(qi, rs)
            f_rand = qf[qi] + qf[jr]; f_rand /= (np.linalg.norm(f_rand) + 1e-12)
            ap_rand_draws.append(per_query_ap(f_rand[None], gf, q_pid[qi:qi+1], q_cam[qi:qi+1],
                                              g_pid, g_cam)[0])
        ap_rand = float(np.mean(ap_rand_draws))
        base_list.append(ap1); union_mean_list.append(ap_mean)
        union_max_list.append(ap_max); rand_union_list.append(ap_rand)

    base_a = np.array(base_list); um = np.array(union_mean_list)
    ux = np.array(union_max_list); ur = np.array(rand_union_list)
    n = len(base_a)
    print(f"\n[3] oracle multi-query on {n} low-support failure queries (mean AP, %):")
    print(f"     single-query (baseline)        AP = {base_a.mean()*100:.3f}")
    print(f"     + same-ID union (MEAN-pool)     AP = {um.mean()*100:.3f}  "
          f"(d={(um.mean()-base_a.mean())*100:+.3f})")
    print(f"     + same-ID union (MAX-pool)      AP = {ux.mean()*100:.3f}  "
          f"(d={(ux.mean()-base_a.mean())*100:+.3f})")
    print(f"     + RANDOM cross-ID union (CTRL)  AP = {ur.mean()*100:.3f}  "
          f"(d={(ur.mean()-base_a.mean())*100:+.3f})  [must NOT recover]")
    # recovery rate: fraction of queries whose AP rises by a meaningful margin
    rec_same = float((um - base_a > 0.05).mean())
    rec_rand = float((ur - base_a > 0.05).mean())
    print(f"     recovery-rate (dAP>+0.05): same-ID={rec_same:.3f}  random-ID={rec_rand:.3f}")

    # CONTROL: k-reciprocal on the SINGLE query (does free re-rank already recover?).
    # Run the VALIDATED re-rank on the FULL query set (real deployment graph context, as P4),
    # then index out our selected low-support failure rows. We compare against base AP computed
    # on the SAME rows via the same dist-matrix path so 'd' is apples-to-apples.
    used_arr = np.array(used)
    print(f"\n[3] k-reciprocal (full-set re-rank, k1={cli.krecip_k1} k2={cli.krecip_k2} "
          f"lam={cli.krecip_lambda}) then index the SAME {n} low-support failure queries:")
    try:
        rr_dist = kreciprocal_rerank(qf, gf, cli.krecip_k1, cli.krecip_k2, cli.krecip_lambda)
        base_dist = 1.0 - qf @ gf.T
        ap_kr = per_query_ap_from_dist(rr_dist, used_arr, q_pid, q_cam, g_pid, g_cam)
        ap_base_rows = per_query_ap_from_dist(base_dist, used_arr, q_pid, q_cam, g_pid, g_cam)
        okkr = (ap_kr >= 0) & (ap_base_rows >= 0)
        d_kr = (ap_kr[okkr].mean() - ap_base_rows[okkr].mean()) * 100
        print(f"     base AP (these rows)= {ap_base_rows[okkr].mean()*100:.3f}  "
              f"(consistency vs single-probe loop base={base_a.mean()*100:.3f})")
        print(f"     k-reciprocal AP     = {ap_kr[okkr].mean()*100:.3f}  (d={d_kr:+.3f} vs base)")
        print(f"     >> same-ID union d={(um.mean()-base_a.mean())*100:+.3f}  vs  "
              f"k-recip d={d_kr:+.3f}  [union is a UNIQUE evidence effect only if union >> k-recip]")
        krecip_mean = float(ap_kr[okkr].mean())
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"     [k-reciprocal failed: {e}]")
        krecip_mean = float('nan')

    return dict(n=n, base=float(base_a.mean()), union_mean=float(um.mean()),
                union_max=float(ux.mean()), rand=float(ur.mean()),
                rec_same=rec_same, rec_rand=rec_rand, krecip=krecip_mean)


# =========================================================================== #
# MAIN
# =========================================================================== #
def main():
    print("#" * 80)
    print(f"# EVIDENCE-SUFFICIENCY KILL-SWITCH  dataset={cli.dataset}  cache={cli.cache_feat}")
    print("#" * 80)
    t0 = time.time()
    q, g = load_data()
    res = per_query_ap(q['feat'], g['feat'], q['pid'], q['cam'], g['pid'], g['cam'])
    print(f"[SANITY] frozen full-gallery mAP={res[res>=0].mean()*100:.2f} nq={int((res>=0).sum())}")

    T1 = test_1(q, g)
    T2 = test_2(q, g)
    T3 = test_3(q, g)

    print("\n" + "#" * 80)
    print(f"# SUMMARY / VERDICT  ({cli.dataset})  [{time.time()-t0:.0f}s]")
    print("#" * 80)
    print("# A signal counts ONLY if it survives the trivial controls (1x-top1-margin + "
          "#false-in-topk). HUBNESS §7.6: high raw rho but partial~0 = NO unique signal = DEAD.")
    print(f"[T1] tax-residual partial (| margin+#false): lowtail={T1['partial_lowtail']:+.3f}  "
          f"#cross-cam-pos={T1['partial_ncc']:+.3f}  dispersion={T1['partial_disp']:+.3f}  "
          f"| OOF incr over trivials={T1['incr']:+.3f}")
    print(f"[T2] failure-AUC: best support-trivial gap={T2['best_supp_minus_triv']:+.3f}  "
          f"OOF incr={T2['incr']:+.3f}  support-solo AUC={T2['supp_solo']:.3f}  "
          f"partial(best-supp,-AP|triv)={T2['partial']:+.3f}")
    print(f"[T3] oracle on n={T3['n']} low-support failures: base={T3['base']*100:.2f} -> "
          f"same-ID-union={T3['union_mean']*100:.2f} (d={(T3['union_mean']-T3['base'])*100:+.2f})  "
          f"random-ID={T3['rand']*100:.2f} (d={(T3['rand']-T3['base'])*100:+.2f})  "
          f"k-recip={T3['krecip']*100:.2f}")
    print(f"[T3] recovery-rate same-ID={T3['rec_same']:.3f} vs random-ID={T3['rec_rand']:.3f}")
    # crude verdict flags
    t1_live = (abs(T1['partial_lowtail']) > 0.10 or abs(T1['partial_ncc']) > 0.10) and T1['incr'] > 0.02
    t2_live = T2['best_supp_minus_triv'] > 0.03 and T2['incr'] > 0.02
    t3_live = (T3['union_mean'] - T3['base']) * 100 > 3.0 and \
              (T3['union_mean'] - T3['base']) > 2 * max(1e-9, T3['rand'] - T3['base'])
    print(f"\n[VERDICT] T1 independent-of-trivial signal: {'LIVE' if t1_live else 'DEAD'}  "
          f"| T2: {'LIVE' if t2_live else 'DEAD'}  "
          f"| T3 evidence-insufficiency: {'LIVE' if t3_live else 'DEAD'}")
    print("# (final call is human; flags are conservative thresholds, read the numbers above)")
    print("[done]")


if __name__ == '__main__':
    main()
