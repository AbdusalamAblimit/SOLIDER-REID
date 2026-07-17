#!/usr/bin/env python3
"""Rank-Regret (Rank-Instability) ReID routing — ZERO-TRAINING kill-switch.

EFFICIENCY re-frame (NOT an accuracy hidden variable — those collide with
k-reciprocal; this is the compute/Pareto axis):

    ReID by default runs the FULL network on every query (uniform compute).
    RE-DEFINE: route compute using the *retrieval-rank disagreement* between a
    CHEAP early-layer feature and the FULL final feature — the Rank-Regret /
    Rank-Instability (RI) of a query. Low RI (cheap ranking == full ranking) ->
    EARLY-EXIT on the cheap feature; high RI -> spend FULL compute.

    KEY (decides life/death; distinguishes from CFPER-style *difficulty* routing):
    RI is a RETRIEVAL-RESULT-LEVEL / RELATION-LEVEL variable (does the cheap
    representation re-order the gallery?), NOT an image-internal difficulty score.
    We MUST show RI predicts the per-query (AP_full - AP_cheap) gap BETTER than
    static difficulty proxies (cheap top1-margin / entropy / feature-norm /
    top1-top2 gap). If RI ~= static difficulty, the re-frame degrades to CFPER
    and is DEAD.

Cheap vs Full
    cheap = GAP over an EARLY Swin stage output (stage2 = featmaps[1], or
            stage3 = featmaps[2]); a low-cost partial forward.
    full  = final BNNeck embedding (the trained eval feature = featmaps[3] GAP
            -> bottleneck). cheap is strictly weaker than full (verified -> headroom).

Tests
    A   RI ~ (AP_full - AP_cheap) per query?  (spearman + perm-p)
    B*  RI vs static difficulty proxies: which better predicts (AP_full-AP_cheap)?
        RI must stay significant in PARTIAL corr controlling all static proxies.
        ** life/death — CFPER collision test **
    C   cheap-ONLY quantities (no full feature) -> RI?  (feasibility: at inference
        we cannot see full; we must estimate RI from cheap signals to route.)
    D   Pareto cascade: low-RI (oracle) -> cheap, high-RI -> full; sweep threshold,
        plot compute-fraction vs mAP. Compare random routing / static-difficulty
        routing / fixed-budget cascade. PASS = ~50-60% compute keeps >=99% full mAP
        AND clearly beats random + static-difficulty routing.

NOTHING is trained: frozen ckpt + torch.no_grad + numpy.

Run (lab-3090-d):
  Market:
    cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 \
      /root/miniconda3/envs/solider-reid/bin/python \
      experiments/cargo_cvpb/cvpb_rankregret_killswitch.py \
      --config configs/market/pose_psg_lgpa_gcn_base.yml \
      --ckpt   log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth \
      --dataset market1501 \
      --cache_feat /tmp/rr_market_feats.npz 2>&1 | tee /tmp/cvpb_rr_market.log
  Occluded-Duke:
    --config configs/occluded_duke/pose_psg_lgpa_gcn512_2stage_small.yml \
    --ckpt   log/occluded_duke/exp255_small_gcn512_2stage/transformer_120.pth \
    --dataset occluded_duke --cache_feat /tmp/rr_od_feats.npz
  smoke first: add  --smoke 300
"""
import os, sys, time, argparse, json
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.abspath(os.path.join(_here, '..', '..'))   # repo root = .../SOLIDER-REID
sys.path.insert(0, _repo)

ap = argparse.ArgumentParser()
ap.add_argument('--config', default='configs/market/pose_psg_lgpa_gcn_base.yml')
ap.add_argument('--ckpt', default='log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth')
ap.add_argument('--dataset', default='market1501', help='label only (headers/cache name)')
ap.add_argument('--cache_feat', default='/tmp/rr_market_feats.npz',
                help='dump/reuse extracted frozen features (cheap stages + full)')
ap.add_argument('--reuse_feat', action='store_true', help='reuse --cache_feat if present')
ap.add_argument('--smoke', type=int, default=0, help='if >0 cap #query for a fast smoke run')
ap.add_argument('--cheap_stage', type=int, default=1,
                help='which Swin stage GAP is the PRIMARY cheap feature, 0-indexed into the 4 '
                     'stage outputs. Swin depths (2,2,18,2) put ~75%% of FLOPs in stage-2, so the '
                     'only exits that SAVE real compute are stage0 (~8%%) and stage1 (~17%%). '
                     'default 1 (finish stage1 -> ~17%% compute, big saving). stage2/3 save ~0.')
ap.add_argument('--ri_k', type=int, default=20, help='top-k for RI@K (rank disagreement window)')
ap.add_argument('--rbo_p', type=float, default=0.9, help='RBO persistence p (top-weighted)')
ap.add_argument('--out_json', default='', help='optional path to dump a machine-readable summary')
ap.add_argument('--seed', type=int, default=42)
cli = ap.parse_args()
np.random.seed(cli.seed)
RNG = np.random.RandomState(cli.seed)


# =========================================================================== #
# 1. FEATURE EXTRACTION  (frozen ckpt; ONE forward -> all stage GAPs + BNNeck)
#    POSE_TEST_FEAT='global' so the model returns (test_feat, featmaps) with a
#    clean single global vector and the 4 intermediate stage feature maps.
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
        'MODEL.POSE_TEST_FEAT', 'global',   # -> eval returns (global_feat, featmaps); gcn branches off
        'TEST.NECK_FEAT', 'after',          # 'after' = BNNeck feature (trained eval feature) for FULL
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
    print(f"[extract] loaded {cli.ckpt}; POSE_TEST_FEAT=global NECK_FEAT=after; "
          f"num_query={num_query}", flush=True)

    use_pose = cfg.MODEL.POSE_ENABLED
    # We extract: FULL = BNNeck embedding (test_feat); CHEAP_s = GAP(featmaps[s]) for s in 0..3.
    # ALSO the RAW (pre-L2-norm) magnitudes of full + each stage GAP, so feature-norm is a
    # REAL static-difficulty proxy (after L2 the norm is ~1 and useless). [Codex finding]
    full_feats = []
    stage_feats = {0: [], 1: [], 2: [], 3: []}
    full_rawnorm = []
    stage_rawnorm = {0: [], 1: [], 2: [], 3: []}
    n_stage = None
    pids, camids, names = [], [], []
    t0 = time.time()
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
            # eval return with POSE_TEST_FEAT='global' is (test_feat, featmaps)
            assert isinstance(out, (tuple, list)) and len(out) >= 2, \
                f"unexpected eval return: {type(out)} len={len(out) if hasattr(out,'__len__') else '?'}"
            test_feat = out[0]
            featmaps = out[1]
            assert torch.is_tensor(test_feat) and test_feat.dim() == 2, \
                f"FULL feat must be (B,D) vector; got {type(test_feat)} {getattr(test_feat,'shape',None)}"
            assert isinstance(featmaps, (list, tuple)) and len(featmaps) >= 2, \
                f"featmaps must be a list of stage maps; got {type(featmaps)}"
            if n_stage is None:
                n_stage = len(featmaps)
                print(f"[extract] #stage feature maps = {n_stage}; "
                      f"shapes = {[tuple(fm.shape) for fm in featmaps]}", flush=True)
            # FULL: raw BNNeck magnitude (proxy) THEN L2-normed embedding (retrieval)
            full_rawnorm.append(test_feat.norm(p=2, dim=1).cpu().numpy().astype(np.float32))
            ff = F.normalize(test_feat, p=2, dim=1)
            full_feats.append(ff.cpu().numpy().astype(np.float32))
            # CHEAP candidates: GAP of each stage map -> raw norm (proxy) + L2 norm (retrieval)
            for s in range(min(4, len(featmaps))):
                fm = featmaps[s]                       # (B, C, H, W)
                g = fm.mean(dim=(2, 3))                # GAP -> (B, C)
                stage_rawnorm[s].append(g.norm(p=2, dim=1).cpu().numpy().astype(np.float32))
                g = F.normalize(g, p=2, dim=1)
                stage_feats[s].append(g.cpu().numpy().astype(np.float32))
            pids.extend([int(x) for x in b_pids])
            camids.extend([int(x) for x in (b_camids_t.tolist())])
            names.extend([os.path.basename(p) for p in img_paths])
            if bi % 20 == 0:
                print(f"  [extract] batch {bi}/{len(val_loader)} ({time.time()-t0:.0f}s)", flush=True)

    full = np.concatenate(full_feats, 0)
    full_rn = np.concatenate(full_rawnorm, 0)
    stages = {s: np.concatenate(stage_feats[s], 0) for s in stage_feats if len(stage_feats[s])}
    stages_rn = {s: np.concatenate(stage_rawnorm[s], 0) for s in stage_rawnorm if len(stage_rawnorm[s])}
    pids = np.asarray(pids); camids = np.asarray(camids); names = np.asarray(names)
    nq = num_query

    def split(arr):
        return arr[:nq], arr[nq:]

    qf_full, gf_full = split(full)
    q = dict(full=qf_full, pid=pids[:nq], cam=camids[:nq], name=names[:nq])
    g = dict(full=gf_full, pid=pids[nq:], cam=camids[nq:], name=names[nq:])
    q['full_rawnorm'], _ = split(full_rn)
    for s in stages:
        qs, gs = split(stages[s])
        q[f'stage{s}'] = qs; g[f'stage{s}'] = gs
        q[f'stage{s}_rawnorm'], _ = split(stages_rn[s])

    print(f"[extract] query={len(q['name'])} gallery={len(g['name'])} "
          f"full_dim={full.shape[1]} stage_dims={ {s: stages[s].shape[1] for s in stages} } "
          f"({time.time()-t0:.0f}s)", flush=True)

    save = dict(q_full=q['full'], q_pid=q['pid'], q_cam=q['cam'], q_name=q['name'],
                g_full=g['full'], g_pid=g['pid'], g_cam=g['cam'], g_name=g['name'],
                q_full_rawnorm=q['full_rawnorm'],
                n_stage=np.array([len(stages)]))
    for s in stages:
        save[f'q_stage{s}'] = q[f'stage{s}']
        save[f'g_stage{s}'] = g[f'stage{s}']
        save[f'q_stage{s}_rawnorm'] = q[f'stage{s}_rawnorm']
    np.savez(cli.cache_feat, **save)
    print(f"[extract] cached -> {cli.cache_feat}", flush=True)
    return q, g, sorted(stages.keys())


# =========================================================================== #
# 2. EVAL  (market/duke protocol: drop same pid&cam junk; drop pid==-1 gallery)
# =========================================================================== #
def per_query_ap(distmat, q_pid, q_cam, g_pid, g_cam):
    """Per-query AP (-1 = no valid positive -> dropped). Junk = same pid & same cam."""
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


def map_from_aps(aps):
    v = aps[aps >= 0]
    return float(v.mean()) * 100 if len(v) else float('nan'), int((aps >= 0).sum())


def eval_full(distmat, q_pid, q_cam, g_pid, g_cam, max_rank=10):
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
    return dict(mAP=float(np.mean(all_AP)) * 100, r1=float(all_cmc[0]) * 100,
                r5=float(all_cmc[4]) * 100, r10=float(all_cmc[9]) * 100, nq=len(all_AP))


# =========================================================================== #
# 3. RANKING / RI@K helpers  (numpy, no scipy)
#    For each query we compare the cheap top-k gallery order vs the full top-k
#    gallery order. RI@K is HIGH when cheap re-orders the gallery (rank regret).
# =========================================================================== #
def topk_order(sim, k):
    """(num_q, k) gallery indices, most-similar first, per query (raw kNN, no junk
    removal — RI is a property of the cheap-vs-full *retrieval ordering*)."""
    k = min(k, sim.shape[1])
    idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(sim.shape[0])[:, None]
    order = np.argsort(-sim[rows, idx], axis=1)
    return idx[rows, order]


def topk_overlap_disagree(tk_a, tk_b):
    """1 - |A_k ∩ B_k| / k  per query (set disagreement of the two top-k sets)."""
    nq, k = tk_a.shape
    out = np.empty(nq)
    for i in range(nq):
        out[i] = 1.0 - len(np.intersect1d(tk_a[i], tk_b[i], assume_unique=True)) / k
    return out


def rbo_disagree(tk_a, tk_b, p=0.9):
    """1 - RBO(top-k) per query (rank-biased overlap; top-weighted, handles
    different items at deep ranks). RBO in [0,1], 1=identical prefix."""
    nq, k = tk_a.shape
    # precompute cumulative agreement A_d = |prefix_d(a) ∩ prefix_d(b)| / d
    out = np.empty(nq)
    weights = np.array([(1 - p) * p ** (d - 1) for d in range(1, k + 1)])
    wsum = weights.sum()
    for i in range(nq):
        a = tk_a[i]; b = tk_b[i]
        sa = set(); sb = set()
        overlap = 0
        agree = np.empty(k)
        for d in range(k):
            xa = a[d]; xb = b[d]
            # add a[d]
            if xa in sb:
                overlap += 1
            sa.add(xa)
            # add b[d]
            if xb in sa:
                overlap += 1
            sb.add(xb)
            if xa == xb:
                # both already counted via membership above; correct double count:
                # when equal, xa was not in sb before (unless dup), xb not in sa before adding xa;
                # the two checks above add 1 (xb in sa after adding xa). keep as is.
                pass
            agree[d] = overlap / (d + 1)
        rbo = float((weights * agree).sum() / wsum)
        out[i] = 1.0 - rbo
    return out


def kendall_tau_disagree(tk_a, tk_b):
    """Per query: 1 - Kendall tau over the UNION of the two top-k lists, using
    rank position (missing items get rank k -> deep/penalized). Returns distance
    in [0,1] (0 = same order)."""
    nq, k = tk_a.shape
    out = np.empty(nq)
    for i in range(nq):
        a = tk_a[i]; b = tk_b[i]
        items = np.union1d(a, b)
        # rank in each list (position); not-present -> k (just past the window)
        ra = {g: r for r, g in enumerate(a)}
        rb = {g: r for r, g in enumerate(b)}
        xa = np.array([ra.get(g, k) for g in items], float)
        xb = np.array([rb.get(g, k) for g in items], float)
        # tau-b via concordant/discordant on pair signs
        n = len(items)
        if n < 2:
            out[i] = 0.0; continue
        c = d = 0
        for u in range(n):
            du = xa[u] - xa[u + 1:]
            dv = xb[u] - xb[u + 1:]
            prod = du * dv
            c += int((prod > 0).sum())
            d += int((prod < 0).sum())
        denom = c + d
        tau = (c - d) / denom if denom > 0 else 1.0   # all ties -> identical
        out[i] = (1.0 - tau) / 2.0                    # map [-1,1]->[1,0] distance
    return out


# =========================================================================== #
# 4. STATIC difficulty proxies (cheap-only OR full-only), for Test B / Test C
# =========================================================================== #
def margin_top12(sim):
    """sim(top1) - sim(top2) per query (large margin = confident/easy)."""
    s2 = np.partition(sim, -2, axis=1)[:, -2:]
    s2.sort(axis=1)
    return s2[:, 1] - s2[:, 0]


def gap_top13(sim):
    """sim(top1) - sim(top3) per query (a wider confidence gap; distinct from top1-top2)."""
    s3 = np.partition(sim, -3, axis=1)[:, -3:]
    s3.sort(axis=1)            # ascending -> [:, -1]=top1, [:, -3]=top3
    return s3[:, -1] - s3[:, 0]


def softmax_entropy(sim, tau=0.1, k=50):
    """entropy of the softmax over the top-k cheap similarities (retrieval ambiguity).
    High entropy = flat similarity profile = ambiguous."""
    k = min(k, sim.shape[1])
    tk = np.partition(sim, -k, axis=1)[:, -k:]
    z = tk / tau
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    return -(p * np.log(p + 1e-12)).sum(axis=1)


def neighbor_density(sim, k=10):
    """mean cosine sim to the top-k gallery neighbors (high = dense neighborhood)."""
    k = min(k, sim.shape[1])
    tk = np.partition(sim, -k, axis=1)[:, -k:]
    return tk.mean(axis=1)


# =========================================================================== #
# 5. STATS (spearman, partial spearman, perm-p) — no scipy
# =========================================================================== #
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
    rho = float((rx * ry).sum() / denom) if denom > 0 else float('nan')
    return rho, len(x)


def partial_spearman(x, y, Z):
    """Spearman partial corr of x,y controlling covariate(s) Z (rank-residuals)."""
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
# 6. FLOPs / compute fractions for Swin stages (analytic).
#    Swin-S/B depths = (2,2,18,2). Per stage cost ~ depth * tokens * dim^2 (MSA+MLP
#    dominated by linear projections; window-attn is ~linear in tokens). tokens
#    halve & dim doubles each downsample, so stage GFLOP shares are fixed by the
#    architecture. We compute the CUMULATIVE compute fraction to *finish* stage s
#    (patch-embed + stages 0..s), normalized so finishing stage 3 = 1.0.
# =========================================================================== #
def swin_stage_compute_fractions(stage_dims, depths):
    """Return cumulative compute fraction after finishing each stage (0..3) plus
    a tiny head (BNNeck) cost folded into the last stage.
    cost(stage i) ~ depth_i * tokens_i * dim_i^2 ; tokens_i = T0 / 4^i (2x2 down).
    T0 cancels in the ratio. dims from the actual feature maps."""
    dims = np.asarray(stage_dims, float)            # e.g. [128,256,512,1024] (base) / [96..] (small)
    depths = np.asarray(depths, float)
    tokens = np.array([1.0 / (4 ** i) for i in range(len(dims))])  # relative token count
    stage_cost = depths * tokens * dims ** 2        # per-stage relative FLOPs
    cum = np.cumsum(stage_cost)
    cum = cum / cum[-1]                             # finishing stage3 = 1.0
    return cum                                       # cum[s] = fraction to finish stage s


# =========================================================================== #
# MAIN
# =========================================================================== #
def main():
    DS = cli.dataset
    print("#" * 84)
    print(f"# RANK-REGRET (rank-instability) EFFICIENCY KILL-SWITCH  dataset={DS}")
    print(f"#   ckpt={cli.ckpt}  cheap_stage={cli.cheap_stage} (0-idx into 4 stage GAPs)  "
          f"RI@K={cli.ri_k}")
    print("#" * 84)

    # ---- features ----
    if cli.reuse_feat and os.path.exists(cli.cache_feat):
        z = np.load(cli.cache_feat, allow_pickle=True)
        q = dict(full=z['q_full'], pid=z['q_pid'], cam=z['q_cam'], name=z['q_name'])
        g = dict(full=z['g_full'], pid=z['g_pid'], cam=z['g_cam'], name=z['g_name'])
        if 'q_full_rawnorm' in z.files:
            q['full_rawnorm'] = z['q_full_rawnorm']
        stage_ids = []
        for s in range(4):
            if f'q_stage{s}' in z.files:
                q[f'stage{s}'] = z[f'q_stage{s}']
                g[f'stage{s}'] = z[f'g_stage{s}']
                if f'q_stage{s}_rawnorm' in z.files:
                    q[f'stage{s}_rawnorm'] = z[f'q_stage{s}_rawnorm']
                stage_ids.append(s)
        print(f"[reuse] features from {cli.cache_feat}: q={len(q['name'])} g={len(g['name'])} "
              f"stages={stage_ids}")
    else:
        q, g, stage_ids = extract_features()

    # ---- drop junk gallery (pid == -1) ----
    keep_g = g['pid'] != -1
    for key in list(g.keys()):
        if isinstance(g[key], np.ndarray) and g[key].shape[0] == keep_g.shape[0]:
            g[key] = g[key][keep_g]
    if cli.smoke > 0:
        for key in list(q.keys()):
            if isinstance(q[key], np.ndarray) and q[key].shape[0] >= cli.smoke:
                q[key] = q[key][:cli.smoke]
        print(f"[SMOKE] capped query -> {len(q['name'])}")

    q_pid, q_cam = q['pid'], q['cam']
    g_pid, g_cam = g['pid'], g['cam']
    Nq, Ng = q['full'].shape[0], g['full'].shape[0]

    def l2(a):
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)

    qf_full = l2(q['full'].astype(np.float32)); gf_full = l2(g['full'].astype(np.float32))
    sim_full = qf_full @ gf_full.T
    dm_full = 1.0 - sim_full

    cs = cli.cheap_stage
    assert f'stage{cs}' in q, f"cheap_stage {cs} not extracted; available {stage_ids}"
    qf_cheap = l2(q[f'stage{cs}'].astype(np.float32)); gf_cheap = l2(g[f'stage{cs}'].astype(np.float32))
    sim_cheap = qf_cheap @ gf_cheap.T
    dm_cheap = 1.0 - sim_cheap

    print(f"[data] Nq={Nq} Ng={Ng}  full_dim={qf_full.shape[1]} cheap_dim={qf_cheap.shape[1]}  "
          f"#q-pids={len(np.unique(q_pid))} #g-pids={len(np.unique(g_pid))}")

    # ======================================================================= #
    # STEP 1 — cheap vs full mAP (confirm cheap is weaker -> headroom exists)
    # ======================================================================= #
    print("\n" + "=" * 84)
    print("STEP 1 — cheap-only vs full mAP (every stage GAP) — need cheap << full")
    print("=" * 84)
    res_full = eval_full(dm_full, q_pid, q_cam, g_pid, g_cam)
    print(f"  FULL  (BNNeck embedding)        mAP={res_full['mAP']:6.2f}  R1={res_full['r1']:6.2f}  "
          f"R5={res_full['r5']:6.2f}  R10={res_full['r10']:6.2f}  (nq={res_full['nq']})")
    stage_maps = {}
    for s in stage_ids:
        qs = l2(q[f'stage{s}'].astype(np.float32)); gs = l2(g[f'stage{s}'].astype(np.float32))
        dms = 1.0 - qs @ gs.T
        rs = eval_full(dms, q_pid, q_cam, g_pid, g_cam)
        stage_maps[s] = rs
        tag = '  <== PRIMARY cheap' if s == cs else ''
        print(f"  stage{s} GAP (dim={qs.shape[1]:4d})         mAP={rs['mAP']:6.2f}  R1={rs['r1']:6.2f}  "
              f"R5={rs['r5']:6.2f}  R10={rs['r10']:6.2f}{tag}")
    res_cheap = stage_maps[cs]
    print(f"\n  >> headroom (full - cheap_stage{cs}) mAP = {res_full['mAP']-res_cheap['mAP']:+.2f}  "
          f"R1 = {res_full['r1']-res_cheap['r1']:+.2f}  "
          f"(need clearly POSITIVE — cheap weaker so routing-to-full can help)")

    # ======================================================================= #
    # per-query AP for cheap & full + the GAP that RI must predict
    # ======================================================================= #
    aps_full = per_query_ap(dm_full, q_pid, q_cam, g_pid, g_cam)
    aps_cheap = per_query_ap(dm_cheap, q_pid, q_cam, g_pid, g_cam)
    valid = (aps_full >= 0) & (aps_cheap >= 0)
    ap_gap = aps_full - aps_cheap                      # >0: full helps this query (route to full)
    print(f"\n  per-query (AP_full - AP_cheap): mean={np.nanmean(ap_gap[valid]):+.4f}  "
          f"frac(full>cheap)={float((ap_gap[valid]>0).mean()):.3f}  "
          f"frac(equal)={float((np.abs(ap_gap[valid])<1e-9).mean()):.3f}  (n={int(valid.sum())})")

    # ======================================================================= #
    # RI@K — cheap vs full top-k ranking disagreement (3 metrics)
    # ======================================================================= #
    print("\n" + "=" * 84)
    print(f"RI@K  (cheap-vs-full top-{cli.ri_k} ranking disagreement; HIGH = cheap re-orders)")
    print("=" * 84)
    K = cli.ri_k
    tk_full = topk_order(sim_full, K)
    tk_cheap = topk_order(sim_cheap, K)
    RI_overlap = topk_overlap_disagree(tk_cheap, tk_full)
    RI_rbo = rbo_disagree(tk_cheap, tk_full, p=cli.rbo_p)
    RI_tau = kendall_tau_disagree(tk_cheap, tk_full)
    print(f"  RI(top-k overlap)  mean={RI_overlap.mean():.4f}  std={RI_overlap.std():.4f}  "
          f"[1 - |cap|/k]")
    print(f"  RI(RBO p={cli.rbo_p})     mean={RI_rbo.mean():.4f}  std={RI_rbo.std():.4f}  "
          f"[1 - rank-biased-overlap]")
    print(f"  RI(Kendall-tau)    mean={RI_tau.mean():.4f}  std={RI_tau.std():.4f}  "
          f"[1 - tau over union]/2")
    # inter-metric agreement (sanity: the 3 RI flavors should rank queries similarly)
    r_ov_rbo, _ = spearman(RI_overlap, RI_rbo)
    r_ov_tau, _ = spearman(RI_overlap, RI_tau)
    print(f"  inter-RI spearman: overlap~RBO={r_ov_rbo:+.3f}  overlap~tau={r_ov_tau:+.3f}")
    RI = {'overlap': RI_overlap, 'rbo': RI_rbo, 'tau': RI_tau}
    RI_main = RI_rbo   # RBO is the top-weighted primary RI

    # ======================================================================= #
    # TEST A — RI predicts (AP_full - AP_cheap)?
    # ======================================================================= #
    print("\n" + "=" * 84)
    print("TEST A — RI ~ (AP_full - AP_cheap) per query  (spearman + perm-p)")
    print("=" * 84)
    print("  [honest caveat] A is PARTLY MECHANICAL: RI=0 (cheap order==full order) forces")
    print("  AP_gap=0 by construction, so a positive A is near-tautological. The REAL test is B")
    print("  (does RI beat STATIC difficulty + survive partialling it) — A is only a sanity floor.")
    A = {}
    for name, ri in RI.items():
        rho, n = spearman(ri[valid], ap_gap[valid])
        p = perm_pvalue(ri[valid], ap_gap[valid], rho, n_perm=1000)
        A[name] = dict(rho=rho, p=p, n=n)
        print(f"  rho(RI[{name:7s}], AP_gap) = {rho:+.4f}  perm-p={p:.4f}  (n={n})  "
              f"[expect POSITIVE: high RI -> full helps more]")

    # ======================================================================= #
    # TEST B (★ life/death) — RI vs STATIC difficulty proxies
    #   Static proxies are computed on the CHEAP retrieval (what an inference-time
    #   router could see) AND on the FULL retrieval (steel-man). RI must beat them
    #   AND stay significant after PARTIAL-controlling all of them.
    # ======================================================================= #
    print("\n" + "=" * 84)
    print("TEST B (★CFPER COLLISION) — RI vs static difficulty proxies @ predicting AP_gap")
    print("=" * 84)
    # static proxies on CHEAP retrieval geometry (what an inference-time router could see)
    cheap_margin = margin_top12(sim_cheap)          # large = easy
    cheap_gap13 = gap_top13(sim_cheap)              # large = easy (wider gap)
    cheap_entropy = softmax_entropy(sim_cheap)      # large = ambiguous
    cheap_density = neighbor_density(sim_cheap)      # large = dense
    # RAW (pre-L2) feature magnitudes — REAL norm proxies (post-L2 norm is ~1 = useless).
    # cheap raw-norm is DEPLOYABLE (computed at the cheap stage); full raw-norm is a steel-man.
    cheap_rawnorm = (q[f'stage{cs}_rawnorm'].astype(np.float64)
                     if f'stage{cs}_rawnorm' in q else np.zeros(Nq))
    full_rawnorm = (q['full_rawnorm'].astype(np.float64)
                    if 'full_rawnorm' in q else np.linalg.norm(q['full'].astype(np.float64), axis=1))
    # full-side proxies: the STRONGEST steel-man (they see the same full ranking AP_gap is
    # computed from). Controlling these is the harshest CFPER test — we INCLUDE them.
    full_margin = margin_top12(sim_full)
    full_entropy = softmax_entropy(sim_full)

    proxies = {
        'cheap_margin(neg)':  -cheap_margin,        # small margin -> harder -> sign to align with gap
        'cheap_top1_top3_gap(neg)': -cheap_gap13,   # genuine top1-top3 gap (distinct from top1-top2)
        'cheap_entropy':       cheap_entropy,
        'cheap_density(neg)': -cheap_density,
        'cheap_rawnorm(neg)': -cheap_rawnorm,       # DEPLOYABLE raw cheap-feature magnitude
        'full_feat_norm(neg)': -full_rawnorm,       # REAL raw full norm (steel-man)
        'full_margin(neg)':   -full_margin,         # full-side steel-man
        'full_entropy':        full_entropy,        # full-side steel-man
    }
    # which proxies are CHEAP-ONLY (deployable at inference, before the full forward)?
    cheap_deployable_proxies = ['cheap_margin(neg)', 'cheap_top1_top3_gap(neg)',
                                'cheap_entropy', 'cheap_density(neg)', 'cheap_rawnorm(neg)']
    print("  -- marginal spearman vs AP_gap (each proxy, best-signed) --")
    rho_RI_main, _ = spearman(RI_main[valid], ap_gap[valid])
    print(f"    RI[rbo]                         rho={rho_RI_main:+.4f}   <== RELATION-LEVEL")
    rho_RI_ov, _ = spearman(RI_overlap[valid], ap_gap[valid])
    rho_RI_tau, _ = spearman(RI_tau[valid], ap_gap[valid])
    print(f"    RI[overlap]                     rho={rho_RI_ov:+.4f}")
    print(f"    RI[tau]                         rho={rho_RI_tau:+.4f}")
    proxy_rhos = {}
    for nm, px in proxies.items():
        r, _ = spearman(px[valid], ap_gap[valid])
        proxy_rhos[nm] = r
        print(f"    {nm:30s}  rho={r:+.4f}   [static difficulty]")
    best_proxy = max(proxy_rhos, key=lambda k: abs(proxy_rhos[k]))
    print(f"  -> strongest static proxy: {best_proxy} (|rho|={abs(proxy_rhos[best_proxy]):.4f}) "
          f"vs RI[rbo] |rho|={abs(rho_RI_main):.4f}")

    # PARTIAL: RI controlling ALL static proxies jointly (the decisive line).
    # Stack EVERY static proxy (cheap geometry + full-side steel-man + feat-norm).
    cov_names = ['cheap_margin(neg)', 'cheap_top1_top3_gap(neg)', 'cheap_entropy',
                 'cheap_density(neg)', 'cheap_rawnorm(neg)', 'full_feat_norm(neg)',
                 'full_margin(neg)', 'full_entropy']
    cov_all = np.column_stack([proxies[nm][valid] for nm in cov_names])
    print(f"\n  -- PARTIAL spearman: RI controlling static proxies (the ★ decisive test) --")
    print(f"     (ALL = {len(cov_names)} proxies stacked: {cov_names})")
    B = {}
    each_show = ['cheap_margin(neg)', 'cheap_entropy', 'cheap_density(neg)', 'full_margin(neg)']
    for name, ri in RI.items():
        pr_each = {}
        for nm in each_show:
            r, _ = partial_spearman(ri[valid], ap_gap[valid], proxies[nm][valid])
            pr_each[nm] = r
        pr_all, nall = partial_spearman(ri[valid], ap_gap[valid], cov_all)
        B[name] = dict(partial_all=pr_all, partial_each=pr_each, n=nall)
        print(f"    RI[{name:7s}] | margin       = {pr_each['cheap_margin(neg)']:+.4f}")
        print(f"    RI[{name:7s}] | entropy      = {pr_each['cheap_entropy']:+.4f}")
        print(f"    RI[{name:7s}] | density      = {pr_each['cheap_density(neg)']:+.4f}")
        print(f"    RI[{name:7s}] | full-margin  = {pr_each['full_margin(neg)']:+.4f}")
        print(f"    RI[{name:7s}] | ALL {len(cov_names)} static = {pr_all:+.4f}  (n={nall})  "
              f"<== must stay clearly !=0 (else CFPER collision)")
    # reverse control: do static proxies survive controlling RI? (asymmetry check)
    print("\n  -- reverse: static proxy partialled on RI[rbo] (asymmetry; want proxy to SHRINK) --")
    for nm in ['cheap_margin(neg)', 'cheap_entropy', 'cheap_density(neg)']:
        r_raw, _ = spearman(proxies[nm][valid], ap_gap[valid])
        r_ctrl, _ = partial_spearman(proxies[nm][valid], ap_gap[valid], RI_main[valid])
        print(f"    {nm:22s} rho {r_raw:+.4f} -> | RI[rbo] {r_ctrl:+.4f}")

    # ======================================================================= #
    # TEST C (feasibility) — can CHEAP-ONLY signals estimate RI? (router input)
    #   At inference we route BEFORE computing full; predictor sees only cheap.
    # ======================================================================= #
    print("\n" + "=" * 84)
    print("TEST C (feasibility) — cheap-ONLY signals predict RI? (needed to route at inference)")
    print("=" * 84)
    cheap_only = {
        'cheap_margin(neg)':  -cheap_margin,
        'cheap_top1_top3_gap(neg)': -cheap_gap13,
        'cheap_entropy':       cheap_entropy,
        'cheap_density(neg)': -cheap_density,
        'cheap_rawnorm(neg)': -cheap_rawnorm,
    }
    C = {}
    for ri_name, ri in RI.items():
        row = {}
        for nm, px in cheap_only.items():
            r, _ = spearman(px[valid], ri[valid])
            row[nm] = r
        # simple linear (rank) multi-predictor R: regress RI-rank on cheap-only proxy ranks
        Xr = np.column_stack([_rank(cheap_only[nm][valid]) for nm in cheap_only])
        Xr = np.column_stack([np.ones(Xr.shape[0]), Xr])
        yr = _rank(ri[valid])
        beta, *_ = np.linalg.lstsq(Xr, yr, rcond=None)
        pred = Xr @ beta
        ss_res = ((yr - pred) ** 2).sum(); ss_tot = ((yr - yr.mean()) ** 2).sum()
        R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        rho_pred, _ = spearman(pred, ri[valid])
        C[ri_name] = dict(per_proxy=row, multi_rho=rho_pred, multi_R2=R2)
        print(f"  RI[{ri_name:7s}]:  margin {row['cheap_margin(neg)']:+.3f}  "
              f"entropy {row['cheap_entropy']:+.3f}  density {row['cheap_density(neg)']:+.3f}  "
              f"|  cheap-only MULTI: spearman(pred,RI)={rho_pred:+.3f}  rank-R2={R2:+.3f}")
    print("  >> if cheap-only spearman/R2 ~ 0, RI is UNPREDICTABLE at inference -> NO efficiency gain.")

    # ======================================================================= #
    # TEST D (Pareto cascade) — route low-RI->cheap, high-RI->full; sweep threshold
    # ======================================================================= #
    print("\n" + "=" * 84)
    print("TEST D — Pareto cascade  (route by RI; compute-fraction vs mAP)")
    print("=" * 84)

    # compute fractions to FINISH each stage (cheap exit at stage cs vs full at stage 3)
    stage_dims = [q[f'stage{s}'].shape[1] for s in stage_ids]
    depths = (2, 2, 18, 2)[:len(stage_dims)]
    cum_frac = swin_stage_compute_fractions(stage_dims, depths)
    cheap_compute = float(cum_frac[cs])     # fraction to finish the cheap stage
    full_compute = 1.0
    print(f"  analytic Swin compute fractions to finish each stage: "
          f"{[f'{c:.3f}' for c in cum_frac]}  (depths={depths}, dims={stage_dims})")
    print(f"  cheap (stage{cs}) compute fraction = {cheap_compute:.3f}; full = 1.000")
    print(f"  (NOTE: the FULL forward SUBSUMES the cheap stem — same early stages — so a")
    print(f"   routed-to-full query pays full=1.0 (NOT cheap+full); a cheap-exit query pays only")
    print(f"   cheap_compute. avg compute = frac_full*1.0 + (1-frac_full)*cheap_compute.)")

    # KEY EQUIVALENCE (huge speedup): per-query AP depends ONLY on that query's own
    # distance row. Routing query i to full vs cheap just selects row i = dm_full[i] or
    # dm_cheap[i]; query i's AP is therefore aps_full[i] (routed) or aps_cheap[i] (exit),
    # with NO re-argsort of the (Nq x Ng) matrix. Validity (>=0) depends only on pid/cam
    # (same positives regardless of feature), so aps_full[i]>=0 <=> aps_cheap[i]>=0 and
    # routing never creates/destroys a valid query. We verify this identity once below.
    def cascade_map(route_to_full_mask):
        """mAP under routing, computed from precomputed per-query APs (exact, fast)."""
        aps_routed = np.where(route_to_full_mask, aps_full, aps_cheap)
        mAP, nq = map_from_aps(aps_routed)
        frac_full = float(route_to_full_mask.mean())
        # full path SUBSUMES the cheap stem (same early stages): routed query pays
        # full_compute (=1.0), cheap-exit pays cheap_compute. avg over queries:
        avg_compute = frac_full * full_compute + (1 - frac_full) * cheap_compute
        return mAP, avg_compute, frac_full, aps_routed

    # one-time identity check: precomputed-AP routing == distmat routing on a random mask
    _chk_mask = RNG.rand(Nq) < 0.5
    _dm_chk = np.where(_chk_mask[:, None], dm_full, dm_cheap)
    _aps_chk = per_query_ap(_dm_chk, q_pid, q_cam, g_pid, g_cam)
    _aps_fast = np.where(_chk_mask, aps_full, aps_cheap)
    _vchk = (_aps_chk >= 0) & (_aps_fast >= 0)
    _maxdiff = float(np.abs(_aps_chk[_vchk] - _aps_fast[_vchk]).max()) if _vchk.any() else 0.0
    _valid_match = bool(((_aps_chk >= 0) == (_aps_fast >= 0)).all())
    print(f"  [identity check] precomputed-AP cascade vs distmat cascade: "
          f"max|dAP|={_maxdiff:.2e}, validity-match={_valid_match}  "
          f"(must be ~0 / True — confirms the fast path is exact)")
    assert _maxdiff < 1e-9 and _valid_match, "cascade fast-path identity FAILED"

    base_full_map, _ = map_from_aps(aps_full)
    base_cheap_map, _ = map_from_aps(aps_cheap)
    print(f"\n  endpoints: ALL-cheap mAP={base_cheap_map:.3f} (compute {cheap_compute:.3f}) | "
          f"ALL-full mAP={base_full_map:.3f} (compute 1.000)")

    # ---- ORACLE routing by RI[rbo]: route the HIGHEST-RI fraction to full ----
    def routing_curve_by_score(score, higher_goes_full=True, label='RI'):
        order = np.argsort(-score) if higher_goes_full else np.argsort(score)
        rows = []
        for frac in np.linspace(0, 1, 21):
            n_full = int(round(frac * Nq))
            mask = np.zeros(Nq, bool)
            if n_full > 0:
                mask[order[:n_full]] = True
            mAP, avg_c, ff, _ = cascade_map(mask)
            rows.append((avg_c, mAP, ff))
        return rows

    print("\n  -- routing by RI[rbo] (ORACLE: route highest-RI queries to full) --")
    rows_RI = routing_curve_by_score(RI_main, higher_goes_full=True, label='RI[rbo]')
    # random routing (mean over 20 seeds)
    def random_curve():
        accum = {}
        for frac in np.linspace(0, 1, 21):
            ms = []
            for sd in range(20):
                rng = np.random.RandomState(1000 + sd)
                n_full = int(round(frac * Nq))
                mask = np.zeros(Nq, bool)
                if n_full > 0:
                    mask[rng.permutation(Nq)[:n_full]] = True
                mAP, avg_c, ff, _ = cascade_map(mask)
                ms.append((avg_c, mAP))
            avg_c = np.mean([m[0] for m in ms]); mAP = np.mean([m[1] for m in ms])
            accum[frac] = (avg_c, mAP)
        return [accum[f] for f in np.linspace(0, 1, 21)]
    rows_rand = random_curve()
    # static-difficulty routing — ORACLE upper bound (best static proxy incl. FULL-side;
    # NOT deployable if best_proxy is full-side, but an upper bound on static routing).
    static_score = proxies[best_proxy]
    s_sign = np.sign(spearman(static_score[valid], ap_gap[valid])[0]) or 1.0  # align to AP_gap
    rows_static = routing_curve_by_score(s_sign * static_score, higher_goes_full=True, label=best_proxy)

    # ---------- DEPLOYABLE baselines & router (cheap-only inputs, CROSS-FITTED) ----------
    # 5-fold cross-fit so a query's routing score is produced by a model that never saw its
    # own (cheap-only -> target) pair. Removes the same-set leakage Codex flagged.
    Xcheap = np.column_stack([_rank(cheap_only[nm]) for nm in cheap_only])  # cheap-only design
    Xcheap = np.column_stack([np.ones(Nq), Xcheap])
    def crossfit_score(target_full):
        """OOF linear (rank) prediction of `target_full` from cheap-only inputs.
        target_full may itself use full info (RI or AP_gap) — that is the TRAINING label;
        the produced score for a held-out query is a pure function of its cheap-only inputs."""
        yhat = np.full(Nq, np.nan)
        idx_valid = np.where(valid)[0]
        kf = np.array_split(RNG.permutation(idx_valid), 5)
        for f in range(5):
            te = kf[f]
            tr = np.concatenate([kf[j] for j in range(5) if j != f])
            yr = _rank(target_full[tr])
            beta, *_ = np.linalg.lstsq(Xcheap[tr], yr, rcond=None)
            yhat[te] = Xcheap[te] @ beta
        # also score invalid queries (not used in eval) via a full-data fit, for completeness
        yr = _rank(target_full[idx_valid]); beta, *_ = np.linalg.lstsq(Xcheap[idx_valid], yr, rcond=None)
        nanm = np.isnan(yhat); yhat[nanm] = Xcheap[nanm] @ beta
        return yhat
    # (i) deployable cheap-est RI: cross-fit cheap-only -> oracle RI[rbo]
    RI_hat = crossfit_score(RI_main)
    rows_feasible = routing_curve_by_score(RI_hat, higher_goes_full=True, label='cheap-estRI(xfit)')
    # (ii) THE FAIR HEAD-TO-HEAD (Codex): cheap-only -> AP_gap directly (a static difficulty
    #      router using the SAME cheap inputs). If this matches/beats cheap-estRI, RI adds
    #      nothing beyond a cheap difficulty ensemble — pure CFPER.
    apgap_full_label = np.where(valid, ap_gap, np.nan)
    APgap_hat = crossfit_score(np.nan_to_num(apgap_full_label, nan=np.nanmean(ap_gap[valid])))
    rows_apgap = routing_curve_by_score(APgap_hat, higher_goes_full=True, label='cheap-static-APgap(xfit)')
    # (iii) single best DEPLOYABLE cheap proxy (canonical CFPER difficulty)
    cdp_rhos = {nm: spearman(proxies[nm][valid], ap_gap[valid])[0] for nm in cheap_deployable_proxies}
    best_cheap_proxy = max(cdp_rhos, key=lambda k: abs(cdp_rhos[k]))
    bcp_sign = np.sign(cdp_rhos[best_cheap_proxy]) or 1.0
    rows_cheapstat = routing_curve_by_score(bcp_sign * proxies[best_cheap_proxy],
                                            higher_goes_full=True, label=best_cheap_proxy)

    # report at target compute fractions
    def interp_map_at(rows, target_c):
        cs_ = np.array([r[0] for r in rows]); ms_ = np.array([r[1] for r in rows])
        o = np.argsort(cs_); cs_, ms_ = cs_[o], ms_[o]
        return float(np.interp(target_c, cs_, ms_))

    # targets within the achievable compute range [cheap_compute, 1.0].
    base_targets = [0.50, 0.60, 0.70]
    targets = [t for t in base_targets if t >= cheap_compute - 1e-6]
    if not targets:
        # cheap exit too expensive (e.g. cheap_stage=2 -> 0.917): the [.5,.6,.7] headline is
        # unreachable. Warn loudly and fall back to evenly-spaced points inside [cheap,1].
        print(f"\n  !! WARNING: cheap(stage{cs}) compute={cheap_compute:.3f} > all default targets "
              f"{base_targets}. Swin stage-{cs} exit saves <{100*(1-cheap_compute):.0f}% compute; the "
              f"~50-60% headline is NOT reachable at this stage. Re-run with a CHEAPER --cheap_stage "
              f"(0 or 1) for the real Pareto story. Falling back to in-range probe points.")
        lo = max(cheap_compute, 0.05)
        targets = [round(float(t), 3) for t in np.linspace(lo + 0.02, 0.98, 3)]
    print(f"\n  Pareto table — mAP at avg-compute in {targets} "
          f"(achievable range [{cheap_compute:.3f}, 1.000]).  ORACLE = uses full info; "
          f"DEPLOY = cheap-only cross-fit:")
    print(f"    {'compute':>8s} | {'RI-oracle':>9s} | {'RIhat-DEPLOY':>12s} | "
          f"{'APgap-DEPLOY':>12s} | {'cheapStat':>10s} | {'static-ORACLE':>13s} | {'random':>8s}")
    print(f"    {'':>8s} | {'(RI rank)':>9s} | {'(xfit RI)':>12s} | "
          f"{'(xfit APgap)':>12s} | {best_cheap_proxy[:10]:>10s} | {best_proxy[:13]:>13s} | {'':>8s}")
    D_table = {}
    for tc in targets:
        if tc < cheap_compute - 1e-6:
            continue
        m_ri = interp_map_at(rows_RI, tc)              # oracle RI ranking
        m_feas = interp_map_at(rows_feasible, tc)      # DEPLOYABLE cheap-est RI (xfit)
        m_apgap = interp_map_at(rows_apgap, tc)        # DEPLOYABLE cheap-static AP-gap (xfit) -- fair foe
        m_cheapstat = interp_map_at(rows_cheapstat, tc)  # single best cheap proxy
        m_stat = interp_map_at(rows_static, tc)        # ORACLE static (may use full-side)
        m_rand = interp_map_at(rows_rand, tc)
        D_table[tc] = dict(ri_oracle=m_ri, ri_deploy=m_feas, apgap_deploy=m_apgap,
                           cheapstat=m_cheapstat, static_oracle=m_stat, random=m_rand)
        print(f"    {tc:8.2f} | {m_ri:9.3f} | {m_feas:12.3f} | {m_apgap:12.3f} | "
              f"{m_cheapstat:10.3f} | {m_stat:13.3f} | {m_rand:8.3f}")
    print(f"    {'1.000':>8s} | {base_full_map:9.3f} | {base_full_map:12.3f} | {base_full_map:12.3f} | "
          f"{base_full_map:10.3f} | {base_full_map:13.3f} | {base_full_map:8.3f}   (all-full ref)")
    print(f"\n    KEY: the FAIR deployable fight is RIhat-DEPLOY vs APgap-DEPLOY vs cheapStat "
          f"(all cheap-only). RI-oracle/static-ORACLE use full info = upper bounds, not deployable.")

    # headline: compute to reach >=99% of full mAP, RI vs random vs static
    def compute_for_target_map(rows, target_map):
        cs_ = np.array([r[0] for r in rows]); ms_ = np.array([r[1] for r in rows])
        o = np.argsort(cs_); cs_, ms_ = cs_[o], ms_[o]
        hit = np.where(ms_ >= target_map)[0]
        return float(cs_[hit[0]]) if len(hit) else float('nan')
    tgt99 = 0.99 * base_full_map
    c99_ri = compute_for_target_map(rows_RI, tgt99)
    c99_feas = compute_for_target_map(rows_feasible, tgt99)
    c99_apgap = compute_for_target_map(rows_apgap, tgt99)
    c99_stat = compute_for_target_map(rows_static, tgt99)
    c99_rand = compute_for_target_map(rows_rand, tgt99)
    print(f"\n  compute to reach 99% of full mAP ({tgt99:.3f}):")
    print(f"    RI-oracle = {c99_ri:.3f} | RIhat-DEPLOY = {c99_feas:.3f} | "
          f"APgap-DEPLOY = {c99_apgap:.3f} | static-ORACLE = {c99_stat:.3f} | random = {c99_rand:.3f}")
    print(f"    (efficiency PASS needs RIhat-DEPLOY clearly < random AND < APgap-DEPLOY; "
          f"all=1.000 means NO router saves compute at 99% mAP)")

    # ======================================================================= #
    # SUMMARY
    # ======================================================================= #
    print("\n" + "#" * 84)
    print(f"SUMMARY / VERDICT  ({DS})")
    print("#" * 84)
    print(f"  cheap(stage{cs}) mAP / full mAP            = {res_cheap['mAP']:.2f} / {res_full['mAP']:.2f}  "
          f"(headroom {res_full['mAP']-res_cheap['mAP']:+.2f})")
    print(f"  A  rho(RI[rbo], AP_gap)                  = {A['rbo']['rho']:+.4f} (perm-p {A['rbo']['p']:.4f}); "
          f"overlap {A['overlap']['rho']:+.3f} tau {A['tau']['rho']:+.3f}")
    print(f"  B  best static proxy rho                 = {proxy_rhos[best_proxy]:+.4f} ({best_proxy})")
    print(f"  B  RI[rbo] rho                           = {rho_RI_main:+.4f}")
    print(f"  B* RI[rbo] partial | ALL static proxies  = {B['rbo']['partial_all']:+.4f}  "
          f"<== CFPER LIFE/DEATH (want clearly !=0)")
    print(f"  C  cheap-only -> RI[rbo] multi spearman  = {C['rbo']['multi_rho']:+.3f} "
          f"(rank-R2 {C['rbo']['multi_R2']:+.3f})  <== FEASIBILITY")
    d_show = None
    if D_table:
        tc_show = min(D_table.keys(), key=lambda t: abs(t - 0.60))
        d = D_table[tc_show]; d_show = (tc_show, d)
        print(f"  D  @{tc_show:.0%} compute: RIhat-DEPLOY {d['ri_deploy']:.2f} vs APgap-DEPLOY "
              f"{d['apgap_deploy']:.2f} vs cheapStat {d['cheapstat']:.2f} vs random {d['random']:.2f} "
              f"| RI-oracle {d['ri_oracle']:.2f} static-oracle {d['static_oracle']:.2f} (full {base_full_map:.2f})")
    print(f"  D  compute@99%full: RIhat-DEPLOY {c99_feas:.3f} | APgap-DEPLOY {c99_apgap:.3f} | "
          f"random {c99_rand:.3f} | RI-oracle {c99_ri:.3f}")
    # ---- automated verdict (STRICT; designed NOT to false-positive) ----
    # B (CFPER life/death) requires ALL of:
    #   (i)  RI partial after controlling EVERY static proxy is non-trivial (>=0.08)
    #   (ii) that partial keeps >=50% of RI's own marginal (RI not mostly explained by static)
    #   (iii) RI's marginal BEATS the best static proxy's marginal (RI is the better predictor)
    part_all = abs(B['rbo']['partial_all'])
    best_static_abs = abs(proxy_rhos[best_proxy])
    b_i = part_all >= 0.08
    b_ii = part_all >= 0.50 * abs(rho_RI_main)
    b_iii = abs(rho_RI_main) >= best_static_abs
    b_alive = bool(b_i and b_ii and b_iii)
    c_alive = bool(abs(C['rbo']['multi_rho']) >= 0.10)
    # D (efficiency): the DEPLOYABLE RI router must (a) clearly beat random AND
    #   (b) beat the deployable cheap-static AP-gap router, at the ~60% point and on compute@99%.
    d_alive = False
    if d_show is not None:
        _, d = d_show
        margin_vs_rand = d['ri_deploy'] - d['random']
        margin_vs_apgap = d['ri_deploy'] - d['apgap_deploy']
        d99_ok = np.isfinite(c99_feas) and (not np.isfinite(c99_rand) or c99_feas < c99_rand - 1e-6) \
                 and (not np.isfinite(c99_apgap) or c99_feas <= c99_apgap + 1e-6)
        d_alive = bool(margin_vs_rand > 0.5 and margin_vs_apgap > 0.5 and d99_ok)
    print(f"\n  >> TEST B (CFPER): partial|all={part_all:.3f}(>=.08:{b_i}) "
          f"keeps>=50%marg:{b_ii} RI>best_static({best_static_abs:.3f}):{b_iii}")
    print(f"     {'PASS — RI survives static control AND beats best static proxy' if b_alive else 'FAIL — RI <= static difficulty (CFPER collision)'}")
    print(f"  >> TEST C (feasible): cheap-only->RI spearman={C['rbo']['multi_rho']:+.3f}  "
          f"{'PASS — RI estimable from cheap (router exists)' if c_alive else 'FAIL — RI not estimable from cheap'}")
    print(f"  >> TEST D (efficiency): deployable RI router "
          f"{'PASS — beats random AND cheap-static AP-gap router' if d_alive else 'FAIL — no compute saving / does not beat random+cheap-static'}")
    print(f"\n  >>> OVERALL: idea is "
          f"{'ALIVE (B & C & D all pass)' if (b_alive and c_alive and d_alive) else 'DEAD'}"
          f" — B(novelty/CFPER)={'pass' if b_alive else 'FAIL'} "
          f"C(feasible)={'pass' if c_alive else 'FAIL'} "
          f"D(efficiency)={'pass' if d_alive else 'FAIL'}")
    print("\n[done] rank-regret kill-switch complete.")

    if cli.out_json:
        summ = dict(
            dataset=DS, cheap_stage=cs, ri_k=int(K),
            cheap_mAP=res_cheap['mAP'], full_mAP=res_full['mAP'],
            stage_mAP={int(s): stage_maps[s]['mAP'] for s in stage_ids},
            cheap_compute=cheap_compute, cum_frac=[float(c) for c in cum_frac],
            testA={k: dict(rho=v['rho'], p=v['p']) for k, v in A.items()},
            testB_proxy_rhos=proxy_rhos, testB_best_proxy=best_proxy,
            testB_best_cheap_proxy=best_cheap_proxy,
            testB_RI_marginal=dict(rbo=rho_RI_main, overlap=rho_RI_ov, tau=rho_RI_tau),
            testB_partial_all={k: B[k]['partial_all'] for k in B},
            testC={k: dict(multi_rho=C[k]['multi_rho'], multi_R2=C[k]['multi_R2']) for k in C},
            testD_table={str(k): v for k, v in D_table.items()},
            testD_compute99=dict(ri_oracle=c99_ri, ri_deploy=c99_feas, apgap_deploy=c99_apgap,
                                 static_oracle=c99_stat, random=c99_rand),
            base_full_map=base_full_map,
            verdict=dict(B_alive=bool(b_alive), C_alive=bool(c_alive), D_alive=bool(d_alive),
                         OVERALL_alive=bool(b_alive and c_alive and d_alive)),
        )
        with open(cli.out_json, 'w') as f:
            json.dump(summ, f, indent=2)
        print(f"[json] summary -> {cli.out_json}")


if __name__ == '__main__':
    main()
