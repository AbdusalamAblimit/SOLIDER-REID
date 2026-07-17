#!/usr/bin/env python3
"""Tax-Aware Distractor Training — ZERO-TRAINING cheap precheck (frozen + numpy).

GATE QUESTION
-------------
Gallery-Growth Tax is LIVE (frozen strong ReID: old-query mAP drops structurally
as same-domain gallery grows 1x->10x; Market -4.43 / OD -12.86). The remedy draft
= Tax-Aware Distractor Training (optimize a gallery-size-conditioned extreme
distractor risk). Before spending ANY training, this precheck answers:

  >> Can an EXTREME-NEGATIVE SURROGATE MARGIN, computed from DEPLOY-VISIBLE info
     (1x gallery + a new distractor pool), predict WHICH queries fail under a
     10x gallery?

GO bar: ROC-AUC(margin -> tax-failure) > 0.75  AND it must CLEARLY BEAT the
trivial difficulty scores (1x top1-margin / #false-in-topk / feature-norm / 1x-AP)
both in raw AUC and in INCREMENTAL value after controlling them. Otherwise the
remedy has no signal beyond trivial difficulty = DEAD (the HUBNESS §7.6 lesson:
M(q) looked great at rho+0.65 but its partial corr collapsed to ~0 once
#false-in-topk was controlled; do NOT repeat that mistake).

DESIGN SOURCE: gallery_growth_method_design.txt §②
  p_softmin = -1/a * logsumexp_{p in P(q)}(-a * s(q,p))     # soft min over positives
  n_extreme =  1/b * logsumexp_{n in N(q)}( b * s(q,n))     # soft max over distractors
  margin    = p_softmin - n_extreme

KEY HONESTY CONSTRAINTS:
  * margin is computed on VISIBLE info only: positives in the 1x core gallery +
    scores against the NEW distractor pool (the images that WOULD be injected).
    It NEVER peeks at the 10x-gallery ranking outcome (the label).
  * tax-failure (the label) and margin use a DISJOINT distractor split so the
    surrogate cannot trivially memorize the exact distractors that beat it.
  * every trivial baseline is ALSO computed from the same visible 1x info.

Reuses the kill-switch's feature cache + eval helpers + core/pool split.
Run on lab-3090-d (cached frozen features, pure numpy, no GPU training):
  cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 python3 \
    experiments/cargo_cvpb/cvpb_gallergrowth_precheck.py \
    --dataset market1501 --cache_feat /tmp/hub_market_feats.npz \
    2>&1 | tee /tmp/precheck_market.log
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
# core/pool split (identical convention to the kill-switch test_A)
ap.add_argument('--core_frac', type=float, default=0.2,
                help='fraction of query-IDs kept as CORE task; rest (+gallery-only IDs) = distractor pool')
ap.add_argument('--core_cap', type=int, default=8,
                help='cap core gallery imgs/ID so the held-out pool can reach 10x')
ap.add_argument('--max_mult', type=float, default=10.0, help='target gallery multiplier for the label')
# surrogate temperatures (design §②). a sharpens soft-min over positives, b soft-max over negs.
ap.add_argument('--a_temp', type=float, default=20.0)
ap.add_argument('--b_temp', type=float, default=20.0)
# tax-failure label definitions
ap.add_argument('--drop_abs', type=float, default=15.0, help='abs AP drop (1x->10x, in %) => tax-failure')
ap.add_argument('--drop_quant', type=float, default=0.30, help='bottom q of dAP also => tax-failure (alt label)')
ap.add_argument('--n_seeds', type=int, default=5, help='resample distractor draws for the 10x label, average AP')
ap.add_argument('--topk', type=int, default=10, help='k for #false-in-topk trivial proxy')
cli = ap.parse_args()
np.random.seed(cli.seed)
RNG = np.random.RandomState(cli.seed)


# =========================================================================== #
# data load (reuse kill-switch cache + normalization)
# =========================================================================== #
def load_data():
    z = np.load(cli.cache_feat, allow_pickle=True)
    q = dict(feat=z['q_feat'], pid=z['q_pid'], cam=z['q_cam'], name=z['q_name'])
    g = dict(feat=z['g_feat'], pid=z['g_pid'], cam=z['g_cam'], name=z['g_name'])
    keep = g['pid'] != -1
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
# per-query AP (Market protocol: drop same pid&cam junk). Works on gallery SUBSETS.
# =========================================================================== #
def per_query_ap(qf, gf, q_pid, q_cam, g_pid, g_cam):
    """Return per-query AP (-1 if no valid positive after junk removal). dist=1-cos."""
    sim = qf @ gf.T
    dm = 1.0 - sim
    nq = dm.shape[0]
    order_all = np.argsort(dm, axis=1)
    aps = np.full(nq, -1.0)
    for i in range(nq):
        order = order_all[i]
        keep = ~((g_pid[order] == q_pid[i]) & (g_cam[order] == q_cam[i]))
        gp = g_pid[order][keep]
        m = (gp == q_pid[i]).astype(np.int32)
        if not m.any():
            continue
        prec = m.cumsum() / (np.arange(len(m)) + 1.0)
        aps[i] = (prec * m).sum() / m.sum()
    return aps


# =========================================================================== #
# ROC-AUC (Mann-Whitney U, tie-aware) — predictor where HIGH = more failure.
# margin LOW => failure, so we feed (-margin) as the "risk" score.
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
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0   # midrank, 1-based
        i = j + 1
    return ranks


def roc_auc(risk, label):
    """AUC that risk (higher = more likely positive) ranks positives above negatives.
    Tie-aware via midranks (== Mann-Whitney U with tie correction)."""
    risk = np.asarray(risk, float); label = np.asarray(label, bool)
    ok = np.isfinite(risk)
    risk, label = risk[ok], label[ok]
    n_pos = int(label.sum()); n_neg = int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan'), n_pos, n_neg
    r = _tied_rank(risk)
    auc = (r[label].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc), n_pos, n_neg


def auc_ci_boot(risk, label, nboot=1000, seed=0):
    """Bootstrap 95% CI for AUC (resample queries)."""
    risk = np.asarray(risk, float); label = np.asarray(label, bool)
    ok = np.isfinite(risk); risk, label = risk[ok], label[ok]
    rs = np.random.RandomState(seed)
    n = len(risk); vals = []
    for _ in range(nboot):
        idx = rs.randint(0, n, n)
        if label[idx].sum() == 0 or (~label[idx]).sum() == 0:
            continue
        a, _, _ = roc_auc(risk[idx], label[idx])
        vals.append(a)
    if not vals:
        return float('nan'), float('nan')
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# =========================================================================== #
# logistic regression (numpy, ridge) — for INCREMENTAL-AUC controls.
# Standardize features, fit by Newton/IRLS, no sklearn dependency.
# =========================================================================== #
def _standardize(X):
    X = np.asarray(X, float)
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-9] = 1.0
    return (X - mu) / sd, mu, sd


def logreg_fit(X, y, l2=1.0, iters=200):
    """IRLS logistic regression with ridge. X already has NO intercept col; we add one
    (unpenalized). Returns (w, b)."""
    Xs, mu, sd = _standardize(X)
    n, d = Xs.shape
    Z = np.column_stack([np.ones(n), Xs])         # col0 = intercept
    w = np.zeros(d + 1)
    R = np.eye(d + 1) * l2; R[0, 0] = 0.0          # don't penalize intercept
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
    """Out-of-fold AUC of a logistic model on feature matrix X predicting y.
    Returns OOF AUC (honest, no in-sample optimism)."""
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


# =========================================================================== #
# soft-min / soft-max surrogate (design §②)
# =========================================================================== #
def softmin_pos(s_pos, a):
    """p_softmin = -1/a * logsumexp(-a * s_pos). -> smooth minimum positive sim."""
    s_pos = np.asarray(s_pos, float)
    if len(s_pos) == 0:
        return np.nan
    m = (-a * s_pos).max()
    lse = m + np.log(np.exp(-a * s_pos - m).sum())
    return -lse / a


def softmax_neg(s_neg, b):
    """n_extreme = 1/b * logsumexp(b * s_neg). -> smooth maximum distractor sim."""
    s_neg = np.asarray(s_neg, float)
    if len(s_neg) == 0:
        return np.nan
    m = (b * s_neg).max()
    lse = m + np.log(np.exp(b * s_neg - m).sum())
    return lse / b


# =========================================================================== #
# MAIN PRECHECK
# =========================================================================== #
def main():
    print("#" * 80)
    print(f"# GALLERY-GROWTH-TAX REMEDY PRECHECK  dataset={cli.dataset}")
    print(f"#   surrogate temps a={cli.a_temp} b={cli.b_temp}  target mult={cli.max_mult}x")
    print("#" * 80)
    t0 = time.time()
    q, g = load_data()
    qf, q_pid, q_cam = q['feat'], q['pid'], q['cam']
    gf, g_pid, g_cam = g['feat'], g['pid'], g['cam']

    # ---- core/pool split (same as kill-switch test_A) -------------------------
    q_ids = np.unique(q_pid)
    g_ids = np.unique(g_pid)
    gallery_only_ids = set(g_ids.tolist()) - set(q_ids.tolist())
    RNG.shuffle(q_ids)
    n_core = max(1, int(round(cli.core_frac * len(q_ids))))
    core_ids = set(q_ids[:n_core].tolist())
    pool_ids = set(q_ids[n_core:].tolist()) | gallery_only_ids
    print(f"[split] #query-IDs={len(q_ids)} core-IDs={len(core_ids)} "
          f"pool(distractor)-IDs={len(pool_ids)} (incl {len(gallery_only_ids)} gallery-only)")

    # CORE queries + capped CORE gallery (keep camera diversity for valid AP)
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
    Ng_core, Npool = len(core_idx), len(pool_idx_all)
    n_add_target = int(round((cli.max_mult - 1.0) * Ng_core))
    max_mult_real = 1.0 + Npool / Ng_core
    print(f"[split] core queries={len(cq_pid)} core gallery={Ng_core} (cap {cli.core_cap}/ID) "
          f"distractor pool imgs={Npool}  max-achievable={max_mult_real:.2f}x")
    if n_add_target > Npool:
        n_add_target = Npool
        print(f"[split] !! pool too small for {cli.max_mult:.0f}x; using full pool "
              f"=> effective {1.0+Npool/Ng_core:.2f}x")

    # ---- DISJOINT distractor split: HALF for the LABEL (10x AP), HALF for the
    #      surrogate/trivial features (deploy-visible new-distractor pool). This
    #      prevents the surrogate from peeking at the exact distractors used to
    #      build the label, and mirrors deployment (you score against an incoming
    #      pool that is NOT the future gallery you'll be judged on). ----
    pool_perm = RNG.permutation(pool_idx_all)
    half = len(pool_perm) // 2
    pool_label = pool_perm[:half]            # builds the 10x growth -> tax-failure label
    pool_visible = pool_perm[half:]          # deploy-visible new distractors -> surrogate
    print(f"[split] distractor pool DISJOINT: label-half={len(pool_label)} "
          f"visible-half={len(pool_visible)}")

    # ---- 1x baseline AP per core query ---------------------------------------
    base_aps = per_query_ap(cqf, gf[core_idx], cq_pid, cq_cam, g_pid[core_idx], g_cam[core_idx])
    valid = base_aps >= 0
    print(f"\n[label] 1x baseline (core only): mAP={base_aps[valid].mean()*100:.3f} "
          f"nq={int(valid.sum())} (Ng={Ng_core})")

    # ---- 10x AP per core query (avg over n_seeds draws from the LABEL half) ----
    n_add_lbl = min(n_add_target, len(pool_label))
    big_mult = 1.0 + n_add_lbl / Ng_core
    aps_10x_runs = []
    for s in range(cli.n_seeds):
        rs = np.random.RandomState(cli.seed + 1000 * s + 7)
        add = pool_label if n_add_lbl >= len(pool_label) else rs.choice(pool_label, n_add_lbl, replace=False)
        gidx = np.concatenate([core_idx, add])
        aps_10x_runs.append(per_query_ap(cqf, gf[gidx], cq_pid, cq_cam, g_pid[gidx], g_cam[gidx]))
    aps_10x = np.mean(np.array(aps_10x_runs), axis=0)
    print(f"[label] {big_mult:.1f}x (label half, {cli.n_seeds} draws avg): "
          f"mAP={aps_10x[valid].mean()*100:.3f}  dmAP={(aps_10x[valid].mean()-base_aps[valid].mean())*100:+.3f}")

    # ---- tax-failure labels (two definitions) --------------------------------
    # NOTE: dAP is heavily zero-inflated (most queries lose ~0, a few crash). A
    # quantile THRESHOLD on dAP collapses (the 30th pct of dAP can BE 0 -> '<= thr'
    # then flags every zero-loss query). So def-QUANT is built by RANK: flag exactly
    # the bottom drop_quant fraction of valid queries by most-negative dAP, with ties
    # broken by a fixed RNG jitter so we never overshoot the target count.
    d_ap = (aps_10x - base_aps) * 100.0                # negative = dropped (in % points)
    lab_abs = valid & (d_ap <= -cli.drop_abs)          # def-1: abs drop >= drop_abs
    vidx = np.where(valid)[0]
    nflag = int(round(cli.drop_quant * len(vidx)))
    jit = np.random.RandomState(cli.seed + 9).rand(len(vidx)) * 1e-9
    order = np.argsort(d_ap[vidx] + jit)               # most negative first
    lab_quant = np.zeros_like(valid)
    lab_quant[vidx[order[:nflag]]] = True              # bottom-nflag by dAP
    thr_q_eff = d_ap[vidx[order[max(0, nflag - 1)]]] if nflag > 0 else float('nan')
    print(f"[label] tax-failure: def-ABS(drop>={cli.drop_abs}) n={int(lab_abs.sum())}/{int(valid.sum())} "
          f"({lab_abs.sum()/max(1,valid.sum())*100:.1f}%)   "
          f"def-QUANT(bottom-{cli.drop_quant:.0%} by RANK, n={int(lab_quant.sum())}, "
          f"cutoff dAP~{thr_q_eff:+.2f})")

    # ---- SURROGATE MARGIN (deploy-visible: 1x core gallery positives + VISIBLE
    #      distractor pool). per query: p_softmin over its valid positives in the
    #      1x core gallery; n_extreme over its sims to the visible distractor pool.
    sim_core = cqf @ gf[core_idx].T                    # core queries x core gallery
    sim_vis = cqf @ gf[pool_visible].T                 # core queries x visible distractors
    p_softmin = np.full(len(cq_pid), np.nan)
    n_extreme = np.full(len(cq_pid), np.nan)
    # trivial proxies (ALL deploy-visible, 1x):
    triv_top1margin = np.full(len(cq_pid), np.nan)     # (best pos sim) - (best distractor sim)
    triv_false_topk = np.full(len(cq_pid), np.nan)     # #distractors out-ranking the 1x positives in top-k
    triv_norm = np.linalg.norm(qf[qsel], axis=1)       # raw query feature norm (pre-normalize)
    # base_aps itself is the "1x-AP" trivial proxy.
    cg_pid = g_pid[core_idx]; cg_cam = g_cam[core_idx]
    for i in range(len(cq_pid)):
        # positives in 1x core gallery, junk-removed (same pid, diff cam)
        keep_pos = (cg_pid == cq_pid[i]) & (cg_cam != cq_cam[i])
        s_pos = sim_core[i][keep_pos]
        s_neg_vis = sim_vis[i]                          # all visible distractors are cross-ID by construction
        if len(s_pos) == 0:
            continue
        p_softmin[i] = softmin_pos(s_pos, cli.a_temp)
        n_extreme[i] = softmax_neg(s_neg_vis, cli.b_temp)
        triv_top1margin[i] = s_pos.max() - s_neg_vis.max()
        # #false-in-topk: among the merged 1x-positives + visible distractors, how many
        # distractors rank above the query's WEAKEST kept positive proxy -> count distractors
        # whose sim exceeds the top-k-th positive. Simpler robust proxy: # visible distractors
        # scoring higher than the best positive (deploy-visible hard-distractor count).
        best_pos = s_pos.max()
        triv_false_topk[i] = int((s_neg_vis > best_pos).sum())
    margin = p_softmin - n_extreme                      # HIGH margin = safe; LOW = fragile

    # risk = -margin (higher risk -> more likely tax-failure)
    feats = {
        'margin(extreme-neg)': -margin,                 # ★ remedy's surrogate
        '1x-top1-margin':      -triv_top1margin,        # trivial
        '#false(>bestpos)':     triv_false_topk,        # trivial (count proxy, HUBNESS killer)
        'feat-norm':           -triv_norm,              # trivial (low norm ~ uncertain)
        '1x-AP':               -base_aps,               # trivial (already-hard at 1x)
        'n_extreme-only':       n_extreme,              # ablation: just the soft-max neg term
        'p_softmin-only':      -p_softmin,              # ablation: just the soft-min pos term
    }

    def report(label_name, lab):
        print("\n" + "=" * 78)
        print(f"# PRECHECK vs tax-failure [{label_name}]  "
              f"(pos={int(lab.sum())} neg={int((valid & ~lab).sum())})")
        print("=" * 78)
        ev = valid                                      # evaluate over all valid core queries
        y = lab[ev].astype(bool)
        print(f"  {'predictor':>22}  {'AUC':>7}  {'95% CI':>16}")
        aucs = {}
        for name, risk in feats.items():
            a, npos, nneg = roc_auc(risk[ev], y)
            lo, hi = auc_ci_boot(risk[ev], y, nboot=800, seed=cli.seed)
            aucs[name] = a
            star = ' <<MARGIN' if name.startswith('margin') else ''
            print(f"  {name:>22}  {a:>7.4f}  [{lo:>6.3f},{hi:>6.3f}]{star}")

        # ---- vs-trivial: incremental AUC (LIFE-OR-DEATH, HUBNESS §7.6) --------
        trivial_names = ['1x-top1-margin', '#false(>bestpos)', 'feat-norm', '1x-AP']
        Xtriv = np.column_stack([feats[n][ev] for n in trivial_names])
        m_risk = feats['margin(extreme-neg)'][ev]
        ok = np.isfinite(m_risk) & np.isfinite(Xtriv).all(1)
        yk = y[ok]
        auc_triv_only = cv_auc_logreg(Xtriv[ok], yk.astype(float), seed=cli.seed)
        auc_triv_plus_margin = cv_auc_logreg(np.column_stack([Xtriv[ok], m_risk[ok]]),
                                             yk.astype(float), seed=cli.seed)
        auc_margin_solo = cv_auc_logreg(m_risk[ok][:, None], yk.astype(float), seed=cli.seed)
        print(f"\n  -- vs-TRIVIAL incremental (5-fold OOF logistic, HUBNESS §7.6 lesson) --")
        print(f"     trivials-only (4 proxies)      OOF-AUC = {auc_triv_only:.4f}")
        print(f"     trivials + margin              OOF-AUC = {auc_triv_plus_margin:.4f}")
        print(f"     margin solo                    OOF-AUC = {auc_margin_solo:.4f}")
        print(f"     >> INCREMENT margin adds on top of trivials = "
              f"{auc_triv_plus_margin - auc_triv_only:+.4f}")
        print(f"     >> margin raw AUC - best single trivial      = "
              f"{aucs['margin(extreme-neg)'] - max(aucs[n] for n in trivial_names):+.4f}")

        # partial spearman: does -margin still rank dAP after controlling each trivial?
        from numpy.linalg import lstsq
        def partial_spear(x, ycont, Z):
            x = np.asarray(x, float); ycont = np.asarray(ycont, float); Z = np.asarray(Z, float)
            if Z.ndim == 1: Z = Z[:, None]
            ok2 = np.isfinite(x) & np.isfinite(ycont) & np.isfinite(Z).all(1)
            x, ycont, Z = x[ok2], ycont[ok2], Z[ok2]
            rx, ry = _tied_rank(x), _tied_rank(ycont)
            Zr = np.column_stack([np.ones(len(x))] + [_tied_rank(Z[:, j]) for j in range(Z.shape[1])])
            res = lambda r: r - Zr @ lstsq(Zr, r, rcond=None)[0]
            ex, ey = res(rx), res(ry)
            den = np.sqrt((ex**2).sum() * (ey**2).sum())
            return float((ex * ey).sum() / den) if den > 0 else float('nan')
        # correlate margin-risk with the CONTINUOUS drop (-dAP), controlling all trivials
        neg_dAP = (-d_ap)[ev]
        ps = partial_spear(m_risk, neg_dAP, Xtriv)
        raw_s_margin = partial_spear(m_risk, neg_dAP, np.ones((len(m_risk), 1)))  # ~ raw spearman
        print(f"     partial Spearman(margin-risk, drop | 4 trivials) = {ps:+.4f}  "
              f"(raw = {raw_s_margin:+.4f})")
        return dict(aucs=aucs, auc_triv_only=auc_triv_only,
                    auc_triv_plus_margin=auc_triv_plus_margin,
                    incr=auc_triv_plus_margin - auc_triv_only,
                    margin_minus_best_triv=aucs['margin(extreme-neg)'] - max(aucs[n] for n in trivial_names),
                    partial=ps)

    # ---- DIAGNOSTIC: how redundant is the surrogate margin with each trivial? ----
    # If margin is just a smoothed copy of 1x-top1-margin, its "incremental" AUC is
    # suspect. Report Spearman between margin-risk and each trivial-risk.
    print("\n" + "-" * 78)
    print("# DIAGNOSTIC: Spearman(margin-risk, trivial-risk) — is margin a duplicate?")
    print("-" * 78)
    ev = valid
    mr = (-margin)[ev]
    for name in ['1x-top1-margin', '#false(>bestpos)', 'feat-norm', '1x-AP',
                 'n_extreme-only', 'p_softmin-only']:
        tr = feats[name][ev]
        ok = np.isfinite(mr) & np.isfinite(tr)
        rx, ry = _tied_rank(mr[ok]), _tied_rank(tr[ok])
        rx -= rx.mean(); ry -= ry.mean()
        rho = float((rx * ry).sum() / (np.sqrt((rx**2).sum() * (ry**2).sum()) + 1e-12))
        print(f"  margin-risk vs {name:>20}: Spearman = {rho:+.4f}")

    r_abs = report('ABS drop>=%.0f' % cli.drop_abs, lab_abs)
    r_quant = report('bottom-%.0f%% dAP' % (cli.drop_quant * 100), lab_quant)

    # =====================================================================
    # VERDICT
    # =====================================================================
    print("\n" + "#" * 80)
    print(f"# VERDICT  ({cli.dataset})  [{time.time()-t0:.0f}s]")
    print("#" * 80)
    for tag, r in [('ABS', r_abs), ('QUANT', r_quant)]:
        m_auc = r['aucs']['margin(extreme-neg)']
        go_auc = m_auc > 0.75
        go_triv = (r['margin_minus_best_triv'] > 0.03) and (r['incr'] > 0.02)
        print(f"[{tag}] margin AUC={m_auc:.3f} (>0.75? {go_auc})  "
              f"margin-best_trivial={r['margin_minus_best_triv']:+.3f}  "
              f"incr-over-trivials={r['incr']:+.3f}  "
              f"=> {'GO' if (go_auc and go_triv) else 'NO-GO'}")
    print("# GO requires BOTH: margin AUC>0.75 AND margin clearly beats trivials "
          "(raw>+0.03 AND OOF-incr>+0.02).")
    print("# HUBNESS §7.6: if margin AUC is high but incr-over-trivials ~0, the trivial "
          "difficulty already solved it -> remedy has NO unique signal -> honest DEAD.")
    print("[done]")


if __name__ == '__main__':
    main()
