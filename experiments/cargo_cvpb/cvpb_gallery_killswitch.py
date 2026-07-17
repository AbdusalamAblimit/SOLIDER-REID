#!/usr/bin/env python3
"""Gallery-Composition ReID re-framing — ZERO-TRAINING kill-switch (3 tests).

Unifying re-frame under test (3 independent codex (lifelong d3 / open-set d9 /
long-tail d10) converged here):

    ReID failure is driven by the GALLERY's COMPOSITION (size / growth / shape),
    NOT only by the query or the (frozen) model. Same embedding, different gallery
    -> different failure.

THREE tests, all on FROZEN features (no backward, numpy only):

  A (d3  Gallery-Growth Tax):
      Fix a CORE query/gallery task (a subset of IDs that have query counterparts).
      Progressively inject SAME-DOMAIN held-out IDs' gallery images as pure
      distractors (never a target for any core query), 1x -> 3x/5x/10x gallery.
      Measure d mAP / d R1 of the (unchanged) core queries as gallery grows.
      KEY: model is frozen (representation does NOT change) -> any drop is a
      "gallery-growth tax", part of what LReID reports as "forgetting".
      CONTROL (Hubness lesson): is the drop just #false-in-topk rising (trivial
      mechanical effect of more distractors), or is there a structural topology
      effect? -> per-query d-AP vs (i) #new-distractors-that-out-rank-the-positive
      (trivial), (ii) random-distractor null (shuffle which IDs are distractors).

  B (d9  Gallery-Size Rejection):
      watchlist (enrolled gallery) size {10,50,100,250,500,full}. query = genuine
      (enrolled) + same-domain held-out impostors (never enrolled). Does the
      impostor max-cosine rise systematically with watchlist size (-> global
      threshold's FPIR drifts)? Compare GLOBAL threshold vs SIZE-CONDITIONED
      (per-size impostor-tail) threshold: DIR@FPIR=1%/5%, FPIR@TPIR=90%.
      CONTROL (trivial trap = "max of N draws grows with N"): a RANDOM-feature /
      ROW-SHUFFLED gallery gives the EVT max-of-N baseline; size-conditioning is
      only interesting if it beats global by MORE than on the random control.

  C (d10 Singleton Merge):
      Zipf gallery: head IDs many imgs, tail IDs singletons. A query of a tail
      identity (held-out, "unknown tail") -> does it false-merge into a HEAD
      prototype, and does false-merge rate rise with head SUPPORT count? Compare
      GLOBAL threshold vs SUPPORT-CALIBRATED threshold at matched known-recall.
      CONTROL (trivial trap = "head has more imgs -> more NN lottery tickets"):
      report false-merge vs support AND vs a degree-matched random-label null;
      support-calibration must help BEYOND the mechanical support effect.

Run on lab-3090-d (reuse the hubness feature caches):
  cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 python3 \
    experiments/cargo_cvpb/cvpb_gallery_killswitch.py \
    --dataset market1501 --cache_feat /tmp/hub_market_feats.npz --reuse_feat \
    2>&1 | tee /tmp/cvpb_gallery_market.log
  # occluded_duke: --dataset occluded_duke --cache_feat /tmp/hub_oduke_feats.npz --reuse_feat
"""
import os, sys, time, argparse
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.abspath(os.path.join(_here, '..', '..'))
sys.path.insert(0, _repo)

ap = argparse.ArgumentParser()
ap.add_argument('--config', default='configs/market/pose_psg_lgpa_gcn_base.yml')
ap.add_argument('--ckpt', default='log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth')
ap.add_argument('--dataset', default='market1501')
ap.add_argument('--cache_feat', default='/tmp/hub_market_feats.npz')
ap.add_argument('--reuse_feat', action='store_true')
ap.add_argument('--seed', type=int, default=42)
# Test A
ap.add_argument('--core_frac', type=float, default=0.2,
                help='fraction of query-IDs kept as CORE task; the rest (+gallery-only IDs) are the distractor pool')
ap.add_argument('--core_cap', type=int, default=8,
                help='cap core gallery imgs/ID so the held-out pool can reach 10x (positives still kept)')
ap.add_argument('--growth', type=float, nargs='+', default=[1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
                help='gallery-size multipliers (1x = core gallery only)')
ap.add_argument('--n_growth_seeds', type=int, default=5,
                help='resample which distractor images are injected, average the curve')
# Test B
ap.add_argument('--watchlist_sizes', type=int, nargs='+', default=[10, 50, 100, 250, 500])
ap.add_argument('--n_watch_seeds', type=int, default=20)
# Test C
ap.add_argument('--zipf_a', type=float, default=1.2, help='Zipf exponent for head support sizes')
ap.add_argument('--n_zipf_seeds', type=int, default=10)
cli = ap.parse_args()
np.random.seed(cli.seed)
RNG = np.random.RandomState(cli.seed)


# =========================================================================== #
# FEATURE EXTRACTION (identical convention to hubness/gopl kill-switches)
# =========================================================================== #
def extract_features():
    import torch
    import torch.nn.functional as F
    from config import cfg
    from datasets import make_dataloader
    from model import make_model
    cfg.merge_from_file(os.path.join(_repo, cli.config))
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
    feats, pids, camids, names = [], [], [], []
    use_pose = cfg.MODEL.POSE_ENABLED
    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            imgs = batch[0].cuda(non_blocking=True)
            b_pids = batch[1]; b_camids_t = batch[3]; b_views = batch[4]; img_paths = batch[5]
            pose_dict = batch[6] if (use_pose and len(batch) > 6) else None
            if pose_dict is not None:
                pose_dict = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                             for k, v in pose_dict.items()}
                out = model(imgs, cam_label=b_camids_t.cuda(), view_label=b_views.cuda(),
                            pose_dict=pose_dict)
            else:
                out = model(imgs, cam_label=b_camids_t.cuda(), view_label=b_views.cuda())
            feat = out[0] if isinstance(out, (tuple, list)) else out
            feat = F.normalize(feat, p=2, dim=1)
            feats.append(feat.cpu().numpy().astype(np.float32))
            pids.extend([int(x) for x in b_pids])
            camids.extend([int(x) for x in (b_camids_t.tolist())])
            names.extend([os.path.basename(p) for p in img_paths])
    feats = np.concatenate(feats, 0)
    pids = np.asarray(pids); camids = np.asarray(camids); names = np.asarray(names)
    q = dict(feat=feats[:num_query], pid=pids[:num_query], cam=camids[:num_query], name=names[:num_query])
    g = dict(feat=feats[num_query:], pid=pids[num_query:], cam=camids[num_query:], name=names[num_query:])
    np.savez(cli.cache_feat,
             q_feat=q['feat'], q_pid=q['pid'], q_cam=q['cam'], q_name=q['name'],
             g_feat=g['feat'], g_pid=g['pid'], g_cam=g['cam'], g_name=g['name'])
    return q, g


# =========================================================================== #
# EVAL helpers (Market protocol: drop same pid&cam junk). Work on SUBSETS of g.
# =========================================================================== #
def per_query_ap_cmc(qf, gf, q_pid, q_cam, g_pid, g_cam, max_rank=10, return_falsecnt=False):
    """Return per-query AP (np.array, -1 if no valid positive), CMC matrix, and
    optionally #false-in-topk (k=max_rank) AFTER junk removal — the trivial proxy
    that killed the hubness diagnostic. Distance = 1 - cosine."""
    sim = qf @ gf.T
    dm = 1.0 - sim
    num_q = dm.shape[0]
    order_all = np.argsort(dm, axis=1)
    aps = np.full(num_q, -1.0)
    cmc = np.zeros((num_q, max_rank))
    false_in_topk = np.full(num_q, -1.0)
    nvalid = 0
    for i in range(num_q):
        order = order_all[i]
        keep = ~((g_pid[order] == q_pid[i]) & (g_cam[order] == q_cam[i]))
        gp = g_pid[order][keep]
        m = (gp == q_pid[i]).astype(np.int32)
        if not m.any():
            continue
        nvalid += 1
        c = m.cumsum(); c[c > 1] = 1
        L = min(max_rank, len(c))            # guard: valid gallery may be < max_rank
        cmc[i, :L] = c[:L]
        prec = m.cumsum() / (np.arange(len(m)) + 1.0)
        aps[i] = (prec * m).sum() / m.sum()
        false_in_topk[i] = int((gp[:max_rank] != q_pid[i]).sum())
    valid = aps >= 0
    if nvalid == 0:                          # guard: no query had a valid positive
        nan = float('nan')
        res = dict(mAP=nan, r1=nan, r5=nan, nq=0)
    else:
        res = dict(mAP=float(aps[valid].mean()) * 100,
                   r1=float(cmc[valid, 0].mean()) * 100,
                   r5=float(cmc[valid, 4].mean()) * 100 if max_rank >= 5 else float('nan'),
                   nq=nvalid)
    if return_falsecnt:
        return res, aps, false_in_topk, valid
    return res, aps, valid


def _tied_rank(v):
    """Average (fractional) ranks with proper TIE handling (Codex finding #2).
    Many of our variables have heavy ties (#false-in-topk=0, attraction=0,
    discrete support); plain double-argsort breaks ties by position and can
    fabricate correlation. This returns midrank for tied groups."""
    v = np.asarray(v, float)
    order = np.argsort(v, kind='mergesort')
    ranks = np.empty(len(v), float)
    sv = v[order]
    i = 0
    n = len(v)
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        # midrank (1-based avg) for the tie group [i..j]
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float('nan'), 0
    rx = _tied_rank(x); ry = _tied_rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx**2).sum() * (ry**2).sum())
    return (float((rx * ry).sum() / den) if den > 0 else float('nan')), len(x)


def partial_spearman(x, y, Z):
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


# =========================================================================== #
# DATA LOAD + SPLIT
# =========================================================================== #
def load_data():
    if cli.reuse_feat and os.path.exists(cli.cache_feat):
        z = np.load(cli.cache_feat, allow_pickle=True)
        q = dict(feat=z['q_feat'], pid=z['q_pid'], cam=z['q_cam'], name=z['q_name'])
        g = dict(feat=z['g_feat'], pid=z['g_pid'], cam=z['g_cam'], name=z['g_name'])
        print(f"[reuse] {cli.cache_feat}: q={len(q['name'])} g={len(g['name'])}", flush=True)
    else:
        q, g = extract_features()
    # drop junk gallery (pid==-1 market distractors)
    keep = g['pid'] != -1
    for k in ('feat', 'pid', 'cam', 'name'):
        g[k] = g[k][keep]
    # L2 normalize (idempotent)
    q['feat'] = q['feat'].astype(np.float32)
    g['feat'] = g['feat'].astype(np.float32)
    q['feat'] /= (np.linalg.norm(q['feat'], axis=1, keepdims=True) + 1e-12)
    g['feat'] /= (np.linalg.norm(g['feat'], axis=1, keepdims=True) + 1e-12)
    return q, g


# =========================================================================== #
# TEST A — GALLERY-GROWTH TAX
# =========================================================================== #
def test_A(q, g):
    print("\n" + "#" * 80)
    print("# TEST A — GALLERY-GROWTH TAX (frozen model, grow gallery with held-out distractors)")
    print("#" * 80)
    qf, q_pid, q_cam = q['feat'], q['pid'], q['cam']
    gf, g_pid, g_cam = g['feat'], g['pid'], g['cam']

    # Split query IDs -> CORE (the task) vs DISTRACTOR-POOL IDs (never a target).
    # A SMALL core (core_frac low) + per-ID core gallery CAP keeps the core gallery small
    # so the held-out pool (rest of query IDs + ALL gallery-only IDs) can reach 10x.
    q_ids = np.unique(q_pid)
    g_ids = np.unique(g_pid)
    gallery_only_ids = set(g_ids.tolist()) - set(q_ids.tolist())   # never a query target
    RNG.shuffle(q_ids)
    n_core = max(1, int(round(cli.core_frac * len(q_ids))))
    core_ids = set(q_ids[:n_core].tolist())
    pool_ids = set(q_ids[n_core:].tolist()) | gallery_only_ids      # held-out query IDs + gallery-only
    print(f"[A] #query-IDs={len(q_ids)}  core-IDs={len(core_ids)}  "
          f"pool(distractor)-IDs={len(pool_ids)} (incl {len(gallery_only_ids)} gallery-only)")

    # CORE task: queries whose pid in core_ids; CORE gallery = gallery imgs of core_ids,
    # capped to <= core_cap imgs/ID (keeps core small & makes 10x reachable; positives
    # still present so AP is well-defined).
    qsel = np.array([p in core_ids for p in q_pid])
    cqf, cq_pid, cq_cam = qf[qsel], q_pid[qsel], q_cam[qsel]
    core_idx_list = []
    for cid in core_ids:
        idx = np.where(g_pid == cid)[0]
        if len(idx) > cli.core_cap:
            # keep camera diversity: take >=1 img from each of as many cameras as fit,
            # then fill the rest randomly -> preserves cross-camera positives for AP.
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
    gpool = np.array([p in pool_ids for p in g_pid])
    pool_idx_all = np.where(gpool)[0]
    Ng_core = len(core_idx)
    Npool = len(pool_idx_all)
    print(f"[A] core queries={len(cq_pid)}  core gallery={Ng_core} (cap {cli.core_cap}/ID)  "
          f"distractor pool imgs={Npool}")
    max_mult = 1.0 + Npool / Ng_core
    print(f"[A] max achievable multiplier with this pool = {max_mult:.2f}x")

    # baseline (1x) once
    base_res, base_aps, base_false, base_valid = per_query_ap_cmc(
        cqf, gf[core_idx], cq_pid, cq_cam, g_pid[core_idx], g_cam[core_idx],
        max_rank=10, return_falsecnt=True)
    print(f"[A] 1x baseline (core only): mAP={base_res['mAP']:.3f} R1={base_res['r1']:.3f} "
          f"nq={base_res['nq']} (Ng={Ng_core})")

    print("\n[A] growth curve (mean over seeds; '--' = multiplier exceeds pool):")
    print(f"  {'mult':>5} {'Ng':>7} {'mAP':>8} {'dmAP':>8} {'R1':>8} {'dR1':>8} "
          f"{'mean#false':>11} {'d#false':>9}")
    curve = []
    for mult in cli.growth:
        n_add = int(round((mult - 1.0) * Ng_core))
        if n_add > Npool:
            print(f"  {mult:>5.1f} {'--':>7}  (needs {n_add} distractors, pool has {Npool})")
            continue
        maps, r1s, dfalse_list, per_seed_aps, per_seed_false = [], [], [], [], []
        for s in range(cli.n_growth_seeds):
            rs = np.random.RandomState(cli.seed + 1000 * s + int(mult * 7))
            add_idx = pool_idx_all if n_add >= Npool else rs.choice(pool_idx_all, n_add, replace=False)
            gidx = np.concatenate([core_idx, add_idx])
            res, aps, false_k, valid = per_query_ap_cmc(
                cqf, gf[gidx], cq_pid, cq_cam, g_pid[gidx], g_cam[gidx],
                max_rank=10, return_falsecnt=True)
            maps.append(res['mAP']); r1s.append(res['r1'])
            per_seed_aps.append(aps); per_seed_false.append(false_k)
        mAP = float(np.mean(maps)); r1 = float(np.mean(r1s))
        # mean #false-in-topk across queries, averaged over seeds (vs 1x baseline)
        mean_false = float(np.nanmean([np.where(f >= 0, f, np.nan) for f in per_seed_false]))
        base_mean_false = float(np.nanmean(np.where(base_false >= 0, base_false, np.nan)))
        curve.append(dict(mult=mult, Ng=len(gidx), mAP=mAP, r1=r1,
                          aps=np.mean(np.array(per_seed_aps), axis=0),
                          false=np.mean(np.array(per_seed_false), axis=0)))
        print(f"  {mult:>5.1f} {len(gidx):>7d} {mAP:>8.3f} {mAP-base_res['mAP']:>+8.3f} "
              f"{r1:>8.3f} {r1-base_res['r1']:>+8.3f} {mean_false:>11.3f} "
              f"{mean_false-base_mean_false:>+9.3f}")

    # ---- CONTROL 1: is the d-AP just #false-in-topk rising? (per-query, max mult) ----
    if len(curve) >= 2:
        big = curve[-1]
        # per-query: AP drop vs increase in #false-in-topk from 1x -> max mult
        d_ap = big['aps'] - base_aps                 # negative = dropped
        d_false = big['false'] - base_false          # positive = more false neighbours
        sel = (base_aps >= 0) & (big['aps'] >= 0) & (base_false >= 0) & (big['false'] >= 0)
        rho_drop_false, n_df = spearman(-d_ap[sel], d_false[sel])
        print(f"\n[A] CONTROL1 (Hubness lesson): per-query AP-DROP vs #false-in-topk INCREASE "
              f"(1x->{big['mult']:.0f}x)")
        print(f"     Spearman(-dAP, d#false) = {rho_drop_false:+.4f}  (n={n_df})  "
              f"[high -> the 'tax' is mostly the trivial mechanical count]")
        # E3: partial corr — does the AP DROP track the gallery growth AFTER regressing
        # out d#false? Here growth is fixed (single max mult), so instead we test whether
        # the drop magnitude has ANY structure beyond d#false via partial of -dAP vs the
        # PER-QUERY #positives (few positives -> more fragile) controlling d#false.
        # (Primary structural test remains the no-new-false subset below.)
        pr_drop, _ = partial_spearman(-d_ap[sel], base_false[sel], d_false[sel])
        print(f"     partial Spearman(-dAP, base#false | d#false) = {pr_drop:+.4f}  "
              f"[structure beyond the count increment]")
        # report: among queries whose TOP-10 #false did NOT change, did AP drop?
        # NOTE (Codex #5): #false-in-topk only counts the TOP-10 window. AP can still drop
        # via distractors landing at ranks 11..K between positives -> this subset shows the
        # tax is NOT fully explained by top-10 count, but the rank-reordering it captures is
        # only PARTIAL evidence of structure. The DECISIVE structural proof is CONTROL2
        # (real vs geometry-destroyed at matched count), not this subset.
        no_new_false = sel & (np.abs(d_false) < 1e-6)
        loose_no_new = sel & (np.abs(d_false) < 0.5)
        if no_new_false.sum() >= 10:
            md = float(np.mean(d_ap[no_new_false])) * 100
            print(f"     among {no_new_false.sum()} queries with no change in TOP-10 #false: "
                  f"mean dAP = {md:+.3f}  [drop beyond top-10 count; PARTIAL structure, see CONTROL2]")
        elif loose_no_new.sum() >= 10:
            md = float(np.mean(d_ap[loose_no_new])) * 100
            print(f"     (strict subset only {no_new_false.sum()}; loose |d#false|<0.5 subset "
                  f"n={loose_no_new.sum()}: mean dAP = {md:+.3f})")
        else:
            print(f"     (only {no_new_false.sum()} strict / {loose_no_new.sum()} loose queries "
                  f"with ~no #false change; inconclusive subset)")

    # ---- CONTROL 2: count-matched geometry-DESTROYED null (A1 fix) ----
    # Same n_add distractors drawn from the SAME held-out pool (matched count AND
    # matched per-feature norm), but each distractor vector is COLUMN-SHUFFLED
    # (independent permutation of its coordinates) -> destroys its real direction
    # while keeping the marginal coordinate distribution; re-normalize to unit norm.
    # This isolates "count" from "real held-out geometry" without injecting any
    # core-identity vectors. If real-drop << shuffled-drop -> structural; if ~equal
    # -> the tax is just the mechanical count.
    if len(curve) >= 2:
        big_mult = curve[-1]['mult']
        n_add = int(round((big_mult - 1.0) * Ng_core))
        n_add = min(n_add, Npool)
        rs = np.random.RandomState(cli.seed + 77)
        add_idx = pool_idx_all if n_add >= Npool else rs.choice(pool_idx_all, n_add, replace=False)
        # real distractors (held-out pool)
        gidx = np.concatenate([core_idx, add_idx])
        res_real, _, _ = per_query_ap_cmc(cqf, gf[gidx], cq_pid, cq_cam,
                                          g_pid[gidx], g_cam[gidx], max_rank=10)
        # geometry-destroyed null: same add_idx features, per-row column shuffle
        add_feat = gf[add_idx].copy()
        for r in range(add_feat.shape[0]):
            rs.shuffle(add_feat[r])               # permute coordinates of THIS distractor
        add_feat /= (np.linalg.norm(add_feat, axis=1, keepdims=True) + 1e-12)
        g_feat_mix = np.concatenate([gf[core_idx], add_feat])
        g_pid_mix = np.concatenate([g_pid[core_idx], g_pid[add_idx]])  # distractor labels: never a core target
        g_cam_mix = np.concatenate([g_cam[core_idx], g_cam[add_idx]])
        res_shuf, _, _ = per_query_ap_cmc(cqf, g_feat_mix, cq_pid, cq_cam,
                                          g_pid_mix, g_cam_mix, max_rank=10)
        assert g_feat_mix.shape[0] == gf[gidx].shape[0], "CONTROL2 count mismatch"
        print(f"\n[A] CONTROL2 (structural vs count, matched n_add={n_add}): at {big_mult:.0f}x")
        print(f"     real held-out distractors:      dmAP={res_real['mAP']-base_res['mAP']:+.3f}")
        print(f"     column-shuffled (geom-destroyed): dmAP={res_shuf['mAP']-base_res['mAP']:+.3f}")
        print(f"     [if real << shuffled, the tax is STRUCTURAL (real neighbours bite "
              f"more than direction-randomized ones); if ~equal, it's just count]")
    return curve


# =========================================================================== #
# TEST B — GALLERY-SIZE REJECTION (open-set FPIR drift)
# =========================================================================== #
def enroll_score(qf_one, enrolled_feat):
    """max cosine of a query against an enrolled gallery (proto = per-image)."""
    return float((enrolled_feat @ qf_one).max())


def dir_at_fpir(genuine_scores, impostor_scores, fpir_target):
    """DIR@FPIR: set threshold so impostor accept-rate (FPIR) = target; report the
    fraction of genuine accepted (DIR). Higher impostor scores -> threshold higher
    -> fewer genuine accepted."""
    g = np.asarray(genuine_scores, float); im = np.asarray(impostor_scores, float)
    if len(im) == 0 or len(g) == 0:
        return float('nan'), float('nan')
    thr = np.quantile(im, 1.0 - fpir_target)   # accept score>thr; FPIR = P(impostor>thr)=target
    dir_ = float((g > thr).mean())
    return dir_, float(thr)


def fpir_at_tpir(genuine_scores, impostor_scores, tpir_target):
    g = np.asarray(genuine_scores, float); im = np.asarray(impostor_scores, float)
    if len(im) == 0 or len(g) == 0:
        return float('nan')
    thr = np.quantile(g, 1.0 - tpir_target)    # accept >=thr keeps tpir_target of genuine
    return float((im > thr).mean())


def test_B(q, g):
    print("\n" + "#" * 80)
    print("# TEST B — GALLERY-SIZE REJECTION (impostor max-score & FPIR drift vs watchlist size)")
    print("#" * 80)
    qf, q_pid, q_cam = q['feat'], q['pid'], q['cam']
    gf, g_pid, g_cam = g['feat'], g['pid'], g['cam']

    # ENROLLABLE IDs = appear in BOTH query and gallery (need a genuine probe + an
    # enrolled gallery template). IMPOSTOR source IDs = gallery-only IDs (never a
    # genuine target) PLUS shared IDs not enrolled in a given trial. Using gallery-only
    # IDs (B5 fix) gives a large held-out impostor pool independent of watchlist size.
    q_ids = set(np.unique(q_pid).tolist())
    g_ids = set(np.unique(g_pid).tolist())
    shared = sorted(q_ids & g_ids)
    gallery_only = sorted(g_ids - q_ids)            # never have a genuine query -> pure impostor src
    RNG.shuffle(shared)
    print(f"[B] enrollable (shared q&g) IDs={len(shared)}  gallery-only impostor-source IDs={len(gallery_only)}")
    # impostor probes need a QUERY image -> impostor IDs must be in query. So impostor
    # IDs = shared-but-not-enrolled (they have query imgs). gallery_only IDs have no
    # query image, so they cannot be impostor PROBES, but we note the asymmetry.
    # We therefore cap watchlist size so a disjoint impostor pool of >= size remains.
    max_enroll = len(shared) // 2                    # keep >= size impostor IDs available
    sizes = sorted(set(s for s in cli.watchlist_sizes if s <= max_enroll))
    if not sizes:
        sizes = [max(10, max_enroll)]
    print(f"[B] max_enroll(=len(shared)//2)={max_enroll}; watchlist sizes tested: {sizes}")

    def run_trials(feat_gallery_provider):
        """Returns dict size -> per-trial lists keyed 'gen'/'imp' (one score per probe),
        AND a per-trial split tag so we can do calibration/evaluation folds.
        out[s] = dict(gen=[(trial, score)...], imp=[(trial, score)...])."""
        out = {s: dict(gen=[], imp=[]) for s in sizes}
        for s in sizes:
            for t in range(cli.n_watch_seeds):
                rs = np.random.RandomState(cli.seed + 13 * s + 101 * t)
                enroll_ids = list(rs.choice(shared, s, replace=False))
                enroll_set = set(enroll_ids)
                ef = feat_gallery_provider(enroll_ids, rs)
                for eid in enroll_ids:
                    qcand = np.where(q_pid == eid)[0]
                    qi = qcand[rs.randint(len(qcand))]
                    out[s]['gen'].append((t, enroll_score(qf[qi], ef)))
                # impostors: shared IDs NOT enrolled (have query imgs), disjoint from enrolled
                imp_ids = [i for i in shared if i not in enroll_set]
                rs.shuffle(imp_ids)
                imp_ids = imp_ids[:s]               # balance count with genuine
                for iid in imp_ids:
                    qcand = np.where(q_pid == iid)[0]
                    qi = qcand[rs.randint(len(qcand))]
                    out[s]['imp'].append((t, enroll_score(qf[qi], ef)))
        return out

    def real_provider(enroll_ids, rs):
        mask = np.isin(g_pid, list(enroll_ids))
        return gf[mask]

    def rand_provider(enroll_ids, rs):
        # EVT max-of-N null: enrolled template = SAME count as real, but each template
        # vector is COLUMN-SHUFFLED (Codex #3 fix: plain random-row sampling could include
        # the genuine query's own ID -> label leak). Column shuffle keeps realistic feature
        # marginals/norm but destroys identity alignment, so NO genuine match is possible
        # and cosines come purely from the max-of-N of count-many decorrelated samples.
        mask = np.isin(g_pid, list(enroll_ids))
        n = int(mask.sum())
        rows = rs.choice(gf.shape[0], n, replace=False)
        tpl = gf[rows].copy()
        for r in range(tpl.shape[0]):
            rs.shuffle(tpl[r])
        tpl /= (np.linalg.norm(tpl, axis=1, keepdims=True) + 1e-12)
        return tpl

    def split_scores(pairs, fold):
        """pairs=[(trial,score)]; fold 'cal'=even trials, 'eval'=odd trials."""
        keep = 0 if fold == 'cal' else 1
        return np.array([sc for (t, sc) in pairs if (t % 2) == keep], float)

    # =====================================================================
    # (1) raw impostor-max drift vs watchlist size: REAL vs RANDOM null
    # =====================================================================
    real = run_trials(real_provider)
    rnd = run_trials(rand_provider)
    print("\n[B] impostor max-cosine vs watchlist size (REAL vs RANDOM-feature null, same counts):")
    print(f"  {'size':>6} {'real_imp_mean':>13} {'rnd_imp_mean':>13} {'real_gen_mean':>13} "
          f"{'#imp_probes':>11}")
    real_imp_means, rnd_imp_means = [], []
    for s in sizes:
        rimp = np.array([sc for _, sc in real[s]['imp']])
        nimp = np.array([sc for _, sc in rnd[s]['imp']])
        rgen = np.array([sc for _, sc in real[s]['gen']])
        real_imp_means.append(rimp.mean()); rnd_imp_means.append(nimp.mean())
        print(f"  {s:>6} {rimp.mean():>13.4f} {nimp.mean():>13.4f} {rgen.mean():>13.4f} {len(rimp):>11d}")
    rho_real, _ = spearman(sizes, real_imp_means)
    rho_rnd, _ = spearman(sizes, rnd_imp_means)
    print(f"  Spearman(size, impostor-max): REAL={rho_real:+.4f}  RANDOM-null={rho_rnd:+.4f}  "
          f"(n={len(sizes)} sizes; both >0 EXPECTED; REAL is interesting only if drift "
          f"shape differs from the max-of-N null)")

    # =====================================================================
    # (2) GLOBAL vs SIZE-CONDITIONED threshold — CALIBRATION/EVAL FOLDS (B1 fix)
    #     Thresholds fit on CAL fold (even trials), metrics measured on EVAL fold
    #     (odd trials). No in-sample optimism. Run for REAL and RANDOM (B2 fix).
    # =====================================================================
    def threshold_eval(scores_by_size, tag):
        # pooled CAL impostors/genuine across sizes -> GLOBAL threshold
        all_imp_cal = np.concatenate([split_scores(scores_by_size[s]['imp'], 'cal') for s in sizes])
        all_gen_cal = np.concatenate([split_scores(scores_by_size[s]['gen'], 'cal') for s in sizes])
        rows = []
        gd1, sd1, gd5, sd5, gf90, sf90 = [], [], [], [], [], []
        for s in sizes:
            imp_cal = split_scores(scores_by_size[s]['imp'], 'cal')
            gen_cal = split_scores(scores_by_size[s]['gen'], 'cal')
            imp_ev = split_scores(scores_by_size[s]['imp'], 'eval')
            gen_ev = split_scores(scores_by_size[s]['gen'], 'eval')
            # GLOBAL thresholds (from pooled CAL) measured on EVAL
            thr_g1 = np.quantile(all_imp_cal, 0.99)      # FPIR=1%
            thr_g5 = np.quantile(all_imp_cal, 0.95)      # FPIR=5%
            d1g = float((gen_ev > thr_g1).mean()); d5g = float((gen_ev > thr_g5).mean())
            thr_gT = np.quantile(all_gen_cal, 0.10)      # TPIR=90%
            f90g = float((imp_ev > thr_gT).mean())
            # SIZE-CONDITIONED thresholds (from THIS size's CAL) measured on EVAL
            thr_s1 = np.quantile(imp_cal, 0.99); thr_s5 = np.quantile(imp_cal, 0.95)
            d1s = float((gen_ev > thr_s1).mean()); d5s = float((gen_ev > thr_s5).mean())
            thr_sT = np.quantile(gen_cal, 0.10)
            f90s = float((imp_ev > thr_sT).mean())
            gd1.append(d1g); sd1.append(d1s); gd5.append(d5g); sd5.append(d5s)
            gf90.append(f90g); sf90.append(f90s)
            rows.append((s, d1g, d1s, d5g, d5s, f90g, f90s, len(imp_ev)))
        return rows, (gd1, sd1, gd5, sd5, gf90, sf90)

    print(f"\n[B] GLOBAL vs SIZE-CONDITIONED rejection — CAL/EVAL FOLDS (out-of-sample). REAL:")
    print(f"  {'size':>6} | {'DIR@FPIR1%':>21} | {'DIR@FPIR5%':>21} | {'FPIR@TPIR90%':>23} | {'#imp_ev':>7}")
    print(f"  {'':>6} | {'glob':>10} {'size-c':>10} | {'glob':>10} {'size-c':>10} | "
          f"{'glob':>11} {'size-c':>11} |")
    rows_real, agg_real = threshold_eval(real, 'real')
    for (s, d1g, d1s, d5g, d5s, f90g, f90s, nimp) in rows_real:
        print(f"  {s:>6} | {d1g:>10.3f} {d1s:>10.3f} | {d5g:>10.3f} {d5s:>10.3f} | "
              f"{f90g:>11.3f} {f90s:>11.3f} | {nimp:>7d}")
    gd1, sd1, gd5, sd5, gf90, sf90 = agg_real
    d_dir1_real = float(np.mean(sd1) - np.mean(gd1))
    d_dir5_real = float(np.mean(sd5) - np.mean(gd5))
    drift_red_real = float(np.std(gf90) - np.std(sf90))   # >0 means size-cond flattens drift
    print(f"  REAL: FPIR@TPIR90 std  global={np.std(gf90):.4f}  size-cond={np.std(sf90):.4f}  "
          f"(drift reduction={drift_red_real:+.4f})")
    print(f"  REAL: mean DIR@FPIR1% d(size-cond - global)={d_dir1_real:+.4f}  "
          f"DIR@FPIR5% d={d_dir5_real:+.4f}")

    print(f"\n[B] same on RANDOM-feature null (max-of-N baseline; size-cond gain here is the "
          f"TRIVIAL floor):")
    rows_rnd, agg_rnd = threshold_eval(rnd, 'rnd')
    gd1r, sd1r, gd5r, sd5r, gf90r, sf90r = agg_rnd
    d_dir1_rnd = float(np.mean(sd1r) - np.mean(gd1r))
    d_dir5_rnd = float(np.mean(sd5r) - np.mean(gd5r))
    drift_red_rnd = float(np.std(gf90r) - np.std(sf90r))
    print(f"  RANDOM: FPIR@TPIR90 std global={np.std(gf90r):.4f} size-cond={np.std(sf90r):.4f} "
          f"(drift reduction={drift_red_rnd:+.4f})")
    print(f"  RANDOM: mean DIR@FPIR1% d={d_dir1_rnd:+.4f}  DIR@FPIR5% d={d_dir5_rnd:+.4f}")

    print(f"\n[B] >> NET (REAL - RANDOM): size-conditioning helps BEYOND max-of-N only if these >0:")
    print(f"     net drift-reduction  = {drift_red_real - drift_red_rnd:+.4f}")
    print(f"     net dDIR@FPIR1%      = {d_dir1_real - d_dir1_rnd:+.4f}")
    print(f"     net dDIR@FPIR5%      = {d_dir5_real - d_dir5_rnd:+.4f}")
    return dict(rho_real=rho_real, rho_rnd=rho_rnd,
                drift_red_real=drift_red_real, drift_red_rnd=drift_red_rnd,
                net_drift=drift_red_real - drift_red_rnd,
                glob_f90_std=float(np.std(gf90)), sc_f90_std=float(np.std(sf90)),
                d_dir1=d_dir1_real - d_dir1_rnd,
                d_dir5=d_dir5_real - d_dir5_rnd)


# =========================================================================== #
# TEST C — SINGLETON MERGE (Zipf gallery, tail false-merge vs head support)
# =========================================================================== #
def test_C(q, g):
    print("\n" + "#" * 80)
    print("# TEST C — SINGLETON MERGE (Zipf gallery: tail query false-merges into head proto)")
    print("#" * 80)
    qf, q_pid, q_cam = q['feat'], q['pid'], q['cam']
    gf, g_pid, g_cam = g['feat'], g['pid'], g['cam']

    g_ids = np.unique(g_pid)
    # count available gallery imgs per ID
    avail = {gid: np.where(g_pid == gid)[0] for gid in g_ids}
    counts = np.array([len(avail[gid]) for gid in g_ids])
    print(f"[C] gallery IDs={len(g_ids)}  imgs/ID: min={counts.min()} med={int(np.median(counts))} "
          f"max={counts.max()}")

    # Build a Zipf gallery: assign target support sizes via Zipf, cap by availability.
    # HEAD IDs get many imgs, TAIL IDs get 1 (singleton). Tail QUERY identities are
    # held out (their gallery has 1 img -> they're "rare"); we test if a tail query
    # merges into a HEAD prototype.
    shared = np.array(sorted(set(np.unique(q_pid).tolist()) & set(g_ids.tolist())))
    RNG.shuffle(shared)
    # designate ~60% as head pool, ~40% as tail pool
    n_head = int(0.6 * len(shared))
    head_ids = shared[:n_head]
    tail_ids = shared[n_head:]
    print(f"[C] head-pool IDs={len(head_ids)}  tail-pool IDs={len(tail_ids)}")

    def build_zipf(rs):
        # head supports ~ Zipf, clamp to [2, available]; tail support = 1 (singleton)
        ranks = np.arange(1, len(head_ids) + 1)
        zipf_w = 1.0 / np.power(ranks, cli.zipf_a)
        # map to support sizes 2..maxsupp
        maxsupp = 12
        supp = 2 + np.round((maxsupp - 2) * (zipf_w - zipf_w.min()) /
                            (zipf_w.max() - zipf_w.min() + 1e-12)).astype(int)
        rs.shuffle(supp)
        g_idx, g_id_list, supp_of = [], [], {}
        for hid, ss in zip(head_ids, supp):
            cand = avail[hid]
            take = min(ss, len(cand))
            sel = rs.choice(cand, take, replace=False)
            g_idx.extend(sel.tolist()); supp_of[hid] = take
        for tid in tail_ids:
            cand = avail[tid]
            sel = rs.choice(cand, 1, replace=False)   # singleton in gallery
            g_idx.extend(sel.tolist()); supp_of[tid] = 1
        return np.array(g_idx), supp_of

    # For each tail identity, a held-out QUERY image. To make the tail a TRUE UNKNOWN
    # (C3 fix), the tail's OWN singleton is REMOVED from the gallery for its own probe,
    # so the nearest neighbour is necessarily cross-ID. "false-merge" = nearest neighbour
    # is a HEAD id (tail wrongly absorbed). We measure attraction vs head support, with
    # numerator AND denominator accumulated over the SAME Zipf draws (E2 fix), and a
    # PER-HEAD-ID partial-correlation (E1: hundreds of points, real statistical power).
    head_supp_bins = [(2, 3), (4, 6), (7, 9), (10, 12)]
    nn_is_head_supp = {b: 0 for b in head_supp_bins}
    bin_headcnt = {b: 0 for b in head_supp_bins}
    bin_imgcnt = {b: 0 for b in head_supp_bins}
    total_tail_probes = 0
    nn_is_head_cnt = 0
    # per-head-ID accumulation across seeds: attraction-count, support-sum, appearances
    headid_attract = {hid: 0 for hid in head_ids}     # times this head was a tail-probe's NN
    headid_supp_sum = {hid: 0 for hid in head_ids}     # sum of its support over seeds it appeared
    headid_napp = {hid: 0 for hid in head_ids}         # #seeds it was in the Zipf gallery (always all)

    for t in range(cli.n_zipf_seeds):
        rs = np.random.RandomState(cli.seed + 31 * t)
        gidx, supp_of = build_zipf(rs)
        zgf = gf[gidx]; zg_pid = g_pid[gidx]
        # accumulate per-bin head population for THIS draw (E2: same seeds as numerator)
        for hid in head_ids:
            s = supp_of[hid]
            headid_supp_sum[hid] += s; headid_napp[hid] += 1
            for b in head_supp_bins:
                if b[0] <= s <= b[1]:
                    bin_headcnt[b] += 1; bin_imgcnt[b] += s
        # tail probes: REMOVE the tail's own singleton -> true unknown
        for tid in tail_ids:
            qcand = np.where(q_pid == tid)[0]
            if len(qcand) == 0:
                continue
            qi = qcand[rs.randint(len(qcand))]
            keep = zg_pid != tid                       # drop tail's own gallery img(s)
            sub_pid = zg_pid[keep]
            sim = zgf[keep] @ qf[qi]
            j = int(np.argmax(sim))                    # nearest neighbour (now necessarily cross-ID)
            nn_pid = sub_pid[j]
            total_tail_probes += 1
            hsupp = supp_of.get(nn_pid, 1)
            is_head = hsupp >= 2
            if is_head:
                nn_is_head_cnt += 1
                headid_attract[nn_pid] += 1
                for b in head_supp_bins:
                    if b[0] <= hsupp <= b[1]:
                        nn_is_head_supp[b] += 1

    nn_is_head_frac = nn_is_head_cnt / max(1, total_tail_probes)
    print(f"[C] total tail probes={total_tail_probes}  NN-is-head fraction={nn_is_head_frac:.3f}")

    # ---- BINNED rates (descriptive; n=4 bins, ρ exact-p floor ~0.08 -> NOT a headline) ----
    print("\n[C] P(tail-probe NN is a HEAD of support s) by support bin (DESCRIPTIVE, n=4 bins):")
    print(f"  {'supp-bin':>10} {'#NN(sum)':>10} {'#headIDs(sum)':>14} {'#imgs(sum)':>11} "
          f"{'rate/headID':>12} {'rate/IMAGE':>11}")
    rates_raw, rates_perimg = [], []
    for b in head_supp_bins:
        nn = nn_is_head_supp[b]
        per_headid = nn / max(1, bin_headcnt[b])
        per_img = nn / max(1, bin_imgcnt[b])
        rates_raw.append(per_headid); rates_perimg.append(per_img)
        print(f"  {str(b):>10} {nn:>10d} {bin_headcnt[b]:>14d} {bin_imgcnt[b]:>11d} "
              f"{per_headid:>12.4f} {per_img:>11.5f}")
    print("  [per-IMAGE rate FLAT across support -> purely mechanical count; RISING -> "
          "heads over-attract disproportionately.]")
    supp_mid = [np.mean(b) for b in head_supp_bins]
    rho_id_bin, _ = spearman(supp_mid, rates_raw)
    rho_img_bin, _ = spearman(supp_mid, rates_perimg)
    print(f"  binned Spearman(support, rate/headID)={rho_id_bin:+.4f} (trivially >0)  "
          f"rate/IMAGE={rho_img_bin:+.4f}  [n=4 bins, descriptive only]")

    # ---- ★HEADLINE (E1): PER-HEAD-ID partial correlation (hundreds of points) ----
    # For each head ID: attraction = #tail-probes that landed on it (summed over seeds);
    # support = its mean support; n_imgs ~ support (same thing here). We test whether
    # attraction RISES WITH SUPPORT AFTER CONTROLLING the image-count (support) itself is
    # circular, so we instead control the trivial mechanical effect by comparing:
    #   (a) Spearman(attraction, support)            -- includes the count effect
    #   (b) Spearman(attraction-per-image, support)  -- count removed; the non-trivial claim
    hid_list = [hid for hid in head_ids if headid_napp[hid] > 0]
    attract = np.array([headid_attract[hid] for hid in hid_list], float)
    mean_supp = np.array([headid_supp_sum[hid] / headid_napp[hid] for hid in hid_list], float)
    attract_per_img = attract / (mean_supp * np.array([headid_napp[hid] for hid in hid_list], float))
    rho_attr_supp, n_h = spearman(mean_supp, attract)
    rho_attrpi_supp, _ = spearman(mean_supp, attract_per_img)
    print(f"\n[C] ★PER-HEAD-ID (n={n_h} head IDs, real power):")
    print(f"     Spearman(support, attraction-count)     = {rho_attr_supp:+.4f}  "
          f"[trivially >0: more imgs = more NN tickets]")
    print(f"     Spearman(support, attraction-PER-IMAGE) = {rho_attrpi_supp:+.4f}  "
          f"[NON-TRIVIAL: >0 means heads over-attract beyond count; ~0 means purely count]")

    # ---- support-calibrated vs global threshold, CAL/EVAL FOLD across Zipf seeds ----
    # (Codex #1 fix: even seeds CALIBRATE head-genuine thresholds; odd seeds EVALUATE
    #  tail->head false-merge -> no in-sample circularity. Codex #4 fix: denominator =
    #  ALL tail probes per eval seed, so this is the OVERALL tail->head false-merge rate,
    #  not the rate conditional on NN-already-head.)
    print("\n[C] support-calibrated vs global threshold (OVERALL tail->head false-merge at "
          "matched head-recall), CAL=even seeds / EVAL=odd seeds:")

    def collect_seed(t):
        rs = np.random.RandomState(cli.seed + 31 * t + 5)
        gidx, supp_of = build_zipf(rs)
        zgf = gf[gidx]; zg_pid = g_pid[gidx]
        hg_scores, hg_supp = [], []
        for hid in head_ids:
            qc = np.where(q_pid == hid)[0]
            if len(qc) == 0:
                continue
            qi = qc[rs.randint(len(qc))]
            sim = zgf @ qf[qi]; own = (zg_pid == hid)
            if own.any():
                hg_scores.append(float(sim[own].max())); hg_supp.append(supp_of[hid])
        # tail probes: record (score, nn-support, is-head, n_tail_probes_total)
        ti_score, ti_supp, ti_ishead = [], [], []
        for tid in tail_ids:
            qc = np.where(q_pid == tid)[0]
            if len(qc) == 0:
                continue
            qi = qc[rs.randint(len(qc))]
            keep = zg_pid != tid
            sub_pid = zg_pid[keep]; sim = zgf[keep] @ qf[qi]
            j = int(np.argmax(sim)); nn_pid = sub_pid[j]
            hs = supp_of.get(nn_pid, 1)
            ti_score.append(float(sim[j])); ti_supp.append(hs); ti_ishead.append(hs >= 2)
        return (np.array(hg_scores), np.array(hg_supp),
                np.array(ti_score), np.array(ti_supp), np.array(ti_ishead, bool))

    # pool CAL (even) and EVAL (odd) seeds
    cal_hg_s, cal_hg_sp = [], []
    ev = []   # list of eval-seed tuples
    for t in range(cli.n_zipf_seeds):
        hg_s, hg_sp, ti_s, ti_sp, ti_h = collect_seed(t)
        if len(hg_s) <= 5 or len(ti_s) <= 5:
            continue
        if t % 2 == 0:
            cal_hg_s.append(hg_s); cal_hg_sp.append(hg_sp)
        else:
            ev.append((ti_s, ti_sp, ti_h))
    if cal_hg_s and ev:
        cal_hg_s = np.concatenate(cal_hg_s); cal_hg_sp = np.concatenate(cal_hg_sp)
        n_levels = 0; n_fallback = 0
        for recall in (0.90, 0.95):
            thr_g = np.quantile(cal_hg_s, 1.0 - recall)        # GLOBAL thr from CAL fold
            # per-support thresholds from CAL fold
            thr_by_s = {}
            for s_level in np.unique(cal_hg_sp):
                gen_s = cal_hg_s[cal_hg_sp == s_level]
                if recall == 0.90:
                    n_levels += 1
                if len(gen_s) < 3:
                    gen_s = cal_hg_s
                    if recall == 0.90:
                        n_fallback += 1
                thr_by_s[s_level] = np.quantile(gen_s, 1.0 - recall)
            # evaluate on EVAL fold; OVERALL rate = #false-merges / #ALL tail probes
            fm_g_num = fm_s_num = den_all = 0
            for (ti_s, ti_sp, ti_h) in ev:
                den_all += len(ti_s)                            # ALL tail probes
                # global: merge accepted if score>thr AND nn is a head
                fm_g_num += int(((ti_s > thr_g) & ti_h).sum())
                # support-calibrated: per-probe threshold by its nn-support
                thr_vec = np.array([thr_by_s.get(sp, thr_g) for sp in ti_sp])
                fm_s_num += int(((ti_s > thr_vec) & ti_h).sum())
            fm_g = fm_g_num / max(1, den_all); fm_s = fm_s_num / max(1, den_all)
            print(f"  head-recall={recall:.2f}: OVERALL false-merge  global={fm_g:.4f}  "
                  f"support-calibrated={fm_s:.4f}  (d={fm_s-fm_g:+.4f}; want NEGATIVE)  "
                  f"[{len(ev)} eval seeds, {den_all} tail probes]")
        print(f"  support-level fallback-to-global fraction (sparse levels) = "
              f"{n_fallback/max(1,n_levels):.3f}  [high -> 'support-calibrated' is mostly global]")
    else:
        print("  (insufficient CAL/EVAL seeds for threshold comparison)")
    return dict(rho_attr_supp=rho_attr_supp, rho_attrpi_supp=rho_attrpi_supp,
                rho_img_bin=rho_img_bin, nn_is_head=nn_is_head_frac, n_head=n_h)


# =========================================================================== #
# MAIN
# =========================================================================== #
def main():
    print("#" * 80)
    print(f"# GALLERY-COMPOSITION KILL-SWITCH  dataset={cli.dataset}  ckpt={cli.ckpt}")
    print("#" * 80)
    t0 = time.time()
    q, g = load_data()
    Nq, Ng = len(q['pid']), len(g['pid'])
    print(f"[data] Nq={Nq} Ng={Ng} dim={q['feat'].shape[1]} "
          f"#q-IDs={len(np.unique(q['pid']))} #g-IDs={len(np.unique(g['pid']))}")
    # sanity full mAP
    res, _, _ = per_query_ap_cmc(q['feat'], g['feat'], q['pid'], q['cam'],
                                 g['pid'], g['cam'], max_rank=10)
    print(f"[SANITY] frozen full-gallery mAP={res['mAP']:.2f} R1={res['r1']:.2f} nq={res['nq']}")

    A = test_A(q, g)
    B = test_B(q, g)
    C = test_C(q, g)

    print("\n" + "#" * 80)
    print(f"# SUMMARY / VERDICT  ({cli.dataset})  [{time.time()-t0:.0f}s]")
    print("#" * 80)
    print("# A signal counts ONLY if it survives the trivial baseline (#false-in-topk / "
          "max-of-N / per-image).")
    print(f"[B] impostor-max drift  Spearman(size): REAL={B['rho_real']:+.3f}  "
          f"RANDOM-null={B['rho_rnd']:+.3f}")
    print(f"[B] size-cond NET gain over max-of-N null: drift-red={B['net_drift']:+.4f}  "
          f"dDIR@FPIR1%={B['d_dir1']:+.4f}  dDIR@FPIR5%={B['d_dir5']:+.4f}  "
          f"(>0 -> non-trivial; <=0 -> trivial EVT)")
    print(f"[C] per-head-ID Spearman(support, attraction)={C['rho_attr_supp']:+.3f} (trivial)  "
          f"PER-IMAGE={C['rho_attrpi_supp']:+.3f} (non-trivial claim)  [n={C['n_head']} IDs]")
    print(f"[C] NN-is-head fraction={C['nn_is_head']:.3f}")
    print("# Test A growth-tax verdict: read CONTROL1 (no-new-false subset dAP) & "
          "CONTROL2 (real vs geom-destroyed dmAP) above.")
    print("[done]")


if __name__ == '__main__':
    main()
