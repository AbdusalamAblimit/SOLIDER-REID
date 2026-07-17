#!/usr/bin/env python3
"""OSAC — Open-Set Spectral Over-Collapse — ZERO-TRAINING kill-switch.

Re-frame under test (OSAC_KILLSWITCH_DESIGN.md):
    People assume tighter same-ID compaction is always better for ReID. The hidden
    variable for OPEN-SET ReID is that SEEN-ID neural collapse goes TOO FAR: late in
    training the representation over-aligns to the seen-ID classifier/prototype
    geometry, intra-class variance collapses, and the transferable identity evidence
    for UNSEEN IDs gets low-rank / anisotropic -- pushed into the low-energy spectral
    TAIL. Test is unseen-ID retrieval; k-reciprocal / camera-aware re-rank only
    RE-ORDER existing distances, they cannot resurrect dimensions training squeezed
    out. Hubness (gallery negative in-degree) is a SYMPTOM of this over-collapse, not
    the root cause.

This script trains NOTHING. For each available checkpoint epoch it freezes the ckpt,
extracts a single clean GLOBAL BNNeck feature per image (POSE_TEST_FEAT='global' ->
the model returns only its global-branch vector; PSG still gates the backbone, so the
feature is the real trained eval feature; NECK_FEAT='after' -> post-BNNeck eval
feature), and computes:

Spectral quantities (per epoch ckpt):
    effective rank  = exp(entropy(lambda_i / sum lambda))   (Roy & Vetterli 2007)
    pr (stable rank)= (sum lambda)^2 / sum lambda^2          (participation ratio)
    top-PC energy   = lambda_1 / sum lambda  (and top-10 cumulative)
    NC1             = tr(S_w)/tr(S_b)  on the *seen* TRAIN identities (collapse meter)
    proto-align     = mean over train classes of cos(class-mean, classifier prototype)
    gallery hubness H_k + query hub mass M(q)   (reuse hubness machinery)

CORE TESTS:
 1. OVER-COLLAPSE TRAJECTORY (needs multiple epochs): late training (ep80->120) the
    loss keeps dropping yet effective rank DROPS / top-PC energy RISES / NC1 DROPS.
 2. COLLAPSE <-> RETRIEVAL FAILURE: per-query AP error ~ (query energy on gallery
    top-PCs / 1 - prototype alignment); partial corr survives camera+norm+margin.
    M(q) ~ query top-PC energy (hubness IS a collapse symptom).
 3. DE-COLLAPSE INTERVENTION (zero training): ABTT (remove top-m gallery PCs, sweep m)
    and ZCA whitening on the embedding; does hubness drop and raw mAP rise?

DESTRUCTIVE CONTROLS (decide life/death):
 D1 random-PC / bottom-PC removal must be WORSE than top-PC removal.
 D2 (MOST CRITICAL) DE-COLLAPSE vs k-reciprocal: after ABTT/whitening lifts raw mAP,
    stack k-reciprocal on top and compare to baseline+k-reciprocal. ABTT MUST retain a
    RESIDUAL gain AFTER k-reciprocal (else it is eaten by re-ranking, same death as
    Hubness). This is the make-or-break line.
 D3 control camera/norm/margin -> top-PC<->AP partial corr must survive.
 D4 over-collapse trajectory must appear on BOTH occluded_duke AND market.

Run on lab-3090-d (occluded_duke, full trajectory exp260 ep20..120):
    cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 python3 \
      experiments/cargo_cvpb/cvpb_osac_killswitch.py \
      --config configs/occluded_duke/pose_psg_lgpa_gcn_base.yml \
      --ckpt_dir log/occluded_duke/exp260_base_gcn512_2stage \
      --dataset occluded_duke \
      --cache_dir /tmp/osac_od 2>&1 | tee /tmp/cvpb_osac_od.log
  market:
      --config configs/market/pose_psg_lgpa_gcn_base.yml \
      --ckpt_dir log/market1501/exp260b_base_gcn512_2stage \
      --dataset market1501 --cache_dir /tmp/osac_mk
  smoke first: add  --epochs 120 --smoke 300 --no_train_feats
"""
import os, sys, time, argparse, json, glob, re
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.abspath(os.path.join(_here, '..', '..'))
sys.path.insert(0, _repo)

ap = argparse.ArgumentParser()
ap.add_argument('--config', default='configs/occluded_duke/pose_psg_lgpa_gcn_base.yml')
ap.add_argument('--ckpt_dir', default='log/occluded_duke/exp260_base_gcn512_2stage',
                help='dir holding transformer_{20,40,...,120}.pth')
ap.add_argument('--dataset', default='occluded_duke', help='label only (headers/cache)')
ap.add_argument('--epochs', type=int, nargs='+', default=[20, 40, 60, 80, 100, 120],
                help='which epoch ckpts to analyze (only those present are used)')
ap.add_argument('--cache_dir', default='/tmp/osac_od',
                help='per-epoch feature cache dir (q/g/train features per epoch)')
ap.add_argument('--reuse_feat', action='store_true', help='reuse cached features if present')
ap.add_argument('--no_train_feats', action='store_true',
                help='skip TRAIN feature extraction (NC1/proto-align unavailable; faster smoke)')
ap.add_argument('--train_cap', type=int, default=8000,
                help='cap #train images used for NC1/proto (subsample for speed/memory)')
ap.add_argument('--smoke', type=int, default=0, help='if >0 cap #query for a fast smoke run')
ap.add_argument('--ks', type=int, nargs='+', default=[5, 10, 20], help='top-k for hubness H_k')
ap.add_argument('--k_main', type=int, default=10, help='which k drives M(q)')
ap.add_argument('--abtt_ms', type=int, nargs='+', default=[1, 2, 3, 5, 8, 12, 16, 24, 32],
                help='ABTT: number of top PCs to remove (sweep)')
ap.add_argument('--final_epoch', type=int, default=120,
                help='epoch used for the cross-sectional Test 2/3 + D1-D3 analysis')
ap.add_argument('--seed', type=int, default=42)
cli = ap.parse_args()
np.random.seed(cli.seed)
RNG = np.random.RandomState(cli.seed)
os.makedirs(cli.cache_dir, exist_ok=True)


# =========================================================================== #
# 1. FEATURE EXTRACTION  (frozen ckpt, POSE_TEST_FEAT='global', single vector)
#    Returns query, gallery (val_loader) AND optionally a capped TRAIN set with
#    labels (train_loader_normal -> for NC1 / prototype alignment on SEEN ids).
# =========================================================================== #
def extract_features(ckpt_path, want_train):
    import torch
    import torch.nn.functional as F
    from config import cfg
    from datasets import make_dataloader
    from model import make_model

    cfg.merge_from_file(os.path.join(_repo, cli.config))
    cfg.merge_from_list([
        'TEST.WEIGHT', ckpt_path,
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
    model.load_param(ckpt_path)
    model = model.cuda().eval()
    use_pose = cfg.MODEL.POSE_ENABLED
    print(f"[extract] {os.path.basename(ckpt_path)}; POSE_TEST_FEAT=global; "
          f"num_query={num_query} num_classes={num_classes}", flush=True)

    def run_loader(loader, cap=0):
        feats, pids, camids = [], [], []
        names = []
        t0 = time.time()
        with torch.no_grad():
            seen = 0
            for bi, batch in enumerate(loader):
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
                seen += imgs.shape[0]
                if bi % 40 == 0:
                    print(f"    [batch {bi}/{len(loader)}] ({time.time()-t0:.0f}s)", flush=True)
                if cap and seen >= cap:
                    break
        return (np.concatenate(feats, 0), np.asarray(pids), np.asarray(camids),
                np.asarray(names))

    feats, pids, camids, names = run_loader(val_loader)
    q = dict(feat=feats[:num_query], pid=pids[:num_query], cam=camids[:num_query],
             name=names[:num_query])
    g = dict(feat=feats[num_query:], pid=pids[num_query:], cam=camids[num_query:],
             name=names[num_query:])
    print(f"[extract] query={len(q['name'])} gallery={len(g['name'])} dim={feats.shape[1]}",
          flush=True)

    tr = None
    if want_train:
        # train_loader_normal yields normal (non-PK-sampled) ordering with labels;
        # cap for speed/memory. Used ONLY for NC1 / prototype alignment (seen IDs).
        tf, tp, tc, tn = run_loader(train_loader_normal, cap=cli.train_cap)
        tr = dict(feat=tf, pid=tp, cam=tc, name=tn)
        print(f"[extract] train(capped)={len(tr['pid'])} #train-ids={len(np.unique(tp))}",
              flush=True)

    # also grab the global classifier prototypes from the ckpt (seen-ID prototypes)
    proto = None
    sd = torch.load(ckpt_path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    if 'classifier.weight' in sd:
        proto = sd['classifier.weight'].cpu().numpy().astype(np.float32)  # (C, D)
        print(f"[extract] classifier prototypes: {proto.shape}", flush=True)
    return q, g, tr, proto


def get_or_extract(epoch, want_train):
    cpath = os.path.join(_repo, cli.ckpt_dir, f'transformer_{epoch}.pth')
    if not os.path.exists(cpath):
        return None
    cache = os.path.join(cli.cache_dir, f'feat_ep{epoch}.npz')
    if cli.reuse_feat and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        q = dict(feat=z['q_feat'], pid=z['q_pid'], cam=z['q_cam'], name=z['q_name'])
        g = dict(feat=z['g_feat'], pid=z['g_pid'], cam=z['g_cam'], name=z['g_name'])
        tr = None
        if 'tr_feat' in z.files and z['tr_feat'].size:
            tr = dict(feat=z['tr_feat'], pid=z['tr_pid'], cam=z['tr_cam'], name=z['tr_name'])
        proto = z['proto'] if ('proto' in z.files and z['proto'].size) else None
        print(f"[reuse] ep{epoch} from {cache}: q={len(q['name'])} g={len(g['name'])} "
              f"train={'-' if tr is None else len(tr['pid'])} proto={None if proto is None else proto.shape}")
        return q, g, tr, proto
    q, g, tr, proto = extract_features(cpath, want_train)
    save = dict(q_feat=q['feat'], q_pid=q['pid'], q_cam=q['cam'], q_name=q['name'],
                g_feat=g['feat'], g_pid=g['pid'], g_cam=g['cam'], g_name=g['name'],
                proto=(proto if proto is not None else np.zeros(0, np.float32)))
    if tr is not None:
        save.update(tr_feat=tr['feat'], tr_pid=tr['pid'], tr_cam=tr['cam'], tr_name=tr['name'])
    else:
        save.update(tr_feat=np.zeros(0, np.float32))
    np.savez(cache, **save)
    print(f"[cache] ep{epoch} -> {cache}")
    return q, g, tr, proto


# =========================================================================== #
# 2. EVAL helpers (occluded_duke / market style: drop same pid&cam junk)
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
# 3. SPECTRAL QUANTITIES
# =========================================================================== #
def covariance_eigvals(X):
    """Eigenvalues (descending, >=0) of the centered covariance of rows of X (n,D)."""
    Xc = X - X.mean(0, keepdims=True)
    # covariance (D,D); D ~ 1024 so this is fine
    C = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    w = np.linalg.eigvalsh(C)            # ascending
    w = np.clip(w[::-1], 0.0, None)      # descending, nonneg
    return w


def effective_rank(eigs):
    """exp(Shannon entropy of normalized eigenvalue distribution) (Roy & Vetterli)."""
    s = eigs.sum()
    if s <= 0:
        return 0.0
    p = eigs / s
    p = p[p > 0]
    H = -(p * np.log(p)).sum()
    return float(np.exp(H))


def stable_rank(eigs):
    """(sum lambda)^2 / sum lambda^2 = participation ratio."""
    num = eigs.sum() ** 2
    den = (eigs ** 2).sum()
    return float(num / den) if den > 0 else 0.0


def topk_energy(eigs, k):
    s = eigs.sum()
    if s <= 0:
        return 0.0
    return float(eigs[:k].sum() / s)


def nc1_ratio(X, y):
    """NC1 = tr(S_w) / tr(S_b). Lower => more within-class collapse.
       X (n,D), y (n,) class labels. S_w within-class scatter, S_b between-class."""
    classes = np.unique(y)
    mu = X.mean(0, keepdims=True)
    Sw_tr = 0.0
    Sb_tr = 0.0
    n = X.shape[0]
    for c in classes:
        Xc = X[y == c]
        if len(Xc) < 2:
            # singletons contribute 0 to S_w; still contribute to S_b
            muc = Xc.mean(0, keepdims=True)
            Sb_tr += len(Xc) * float(((muc - mu) ** 2).sum())
            continue
        muc = Xc.mean(0, keepdims=True)
        Sw_tr += float(((Xc - muc) ** 2).sum())
        Sb_tr += len(Xc) * float(((muc - mu) ** 2).sum())
    Sw_tr /= n
    Sb_tr /= n
    return (Sw_tr / Sb_tr) if Sb_tr > 0 else float('nan'), Sw_tr, Sb_tr


def proto_alignment(X, y, proto):
    """Mean cos(class-mean, its classifier prototype) over seen classes present in X.
       proto (C,D) rows are class prototypes indexed by training label id."""
    if proto is None:
        return float('nan'), 0
    C = proto.shape[0]
    P = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-12)
    cos = []
    for c in np.unique(y):
        if c < 0 or c >= C:
            continue
        muc = X[y == c].mean(0)
        muc = muc / (np.linalg.norm(muc) + 1e-12)
        cos.append(float(muc @ P[c]))
    return (float(np.mean(cos)) if cos else float('nan')), len(cos)


# =========================================================================== #
# 4. HUBNESS machinery (ported from cvpb_hubness_killswitch.py)
# =========================================================================== #
def topk_per_query(sim, k):
    idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(sim.shape[0])[:, None]
    order = np.argsort(-sim[rows, idx], axis=1)
    return idx[rows, order]


def compute_Hk(sim, q_pid, g_pid, k, signed='neg'):
    tk = topk_per_query(sim, k)
    H = np.zeros(sim.shape[1], dtype=np.int64)
    for col in range(k):
        gj = tk[:, col]
        if signed == 'all':
            np.add.at(H, gj, 1)
        else:
            same = (g_pid[gj] == q_pid)
            sel = ~same if signed == 'neg' else same
            np.add.at(H, gj[sel], 1)
    return H, tk


def query_hub_mass(tk, H, q_pid, g_pid):
    M = np.zeros(tk.shape[0], dtype=np.float64)
    for col in range(tk.shape[1]):
        gj = tk[:, col]
        neg = (g_pid[gj] != q_pid)
        M += np.where(neg, H[gj], 0.0)
    return M


def skewness(x):
    x = np.asarray(x, float)
    m = x.mean(); s = x.std()
    if s == 0:
        return 0.0
    return float(((x - m) ** 3).mean() / (s ** 3))


# =========================================================================== #
# 5. STATS helpers (no scipy)
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
    x = np.asarray(x, float); y = np.asarray(y, float); Z = np.asarray(Z, float)
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
# 6. DE-COLLAPSE TRANSFORMS  (all unsupervised, fit on GALLERY only, test-time)
# =========================================================================== #
def fit_pca(G):
    """Fit mean + principal axes on gallery rows G (Ng,D). Returns mean, V (D,D)
       eigenvectors as COLUMNS in DESCENDING eigenvalue order, and eigvals."""
    mu = G.mean(0, keepdims=True)
    Gc = G - mu
    C = (Gc.T @ Gc) / max(1, Gc.shape[0] - 1)
    w, V = np.linalg.eigh(C)           # ascending
    order = np.argsort(w)[::-1]
    return mu, V[:, order], np.clip(w[order], 0.0, None)


def abtt_remove(X, mu, V, which):
    """Remove a chosen set of PC directions (columns of V indexed by `which`) from X.
       'all-but-the-top' / ABTT: X' = (X-mu) - sum_{j in which} ((X-mu).v_j) v_j.
       Re-L2-normalize afterwards (cosine retrieval)."""
    Xc = X - mu
    if len(which):
        Vk = V[:, which]                       # (D, |which|)
        proj = (Xc @ Vk) @ Vk.T                # (n, D)
        Xc = Xc - proj
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    return Xn


def zca_whiten(Xq, Xg, mu, V, eigs, eps=1e-3, n_keep=None):
    """ZCA-ish whitening using gallery PCA: rotate to PC space, scale by 1/sqrt(lam+eps),
       rotate back; then L2-normalize. n_keep caps to the top components (dropping the
       deep noise tail). Fit purely on gallery (no labels)."""
    if n_keep is None:
        n_keep = (eigs > 1e-10).sum()
    Vk = V[:, :n_keep]
    s = 1.0 / np.sqrt(eigs[:n_keep] + eps)
    def tf(X):
        Xc = X - mu
        Z = (Xc @ Vk) * s[None, :]             # whitened coords
        Xw = Z @ Vk.T                           # back to D-space (ZCA)
        return Xw / (np.linalg.norm(Xw, axis=1, keepdims=True) + 1e-12)
    return tf(Xq), tf(Xg)


# --------------------------------------------------------------------------- #
# k-reciprocal re-ranking (Zhong 2017), numpy. Returns (Nq,Ng) re-ranked DIST.
# (ported verbatim from cvpb_hubness_killswitch.py, camera_aware optional)
# --------------------------------------------------------------------------- #
def kreciprocal_rerank(qf, gf, q_cam, g_cam, k1=20, k2=6, lam=0.3, camera_aware=False):
    qf = qf.astype(np.float32); gf = gf.astype(np.float32)
    Nq, Ng = qf.shape[0], gf.shape[0]
    allf = np.concatenate([qf, gf], 0)
    cams = np.concatenate([q_cam, g_cam], 0)
    orig = 2.0 - 2.0 * (allf @ allf.T)
    orig = np.maximum(orig, 0.0)
    N = Nq + Ng
    initial_rank = np.argsort(orig, axis=1).astype(np.int32)
    V = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        fwd = initial_rank[i, :k1 + 1]
        recip = []
        for cand in fwd:
            cand_fwd = initial_rank[cand, :k1 + 1]
            if i in cand_fwd:
                recip.append(cand)
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
# helper: hubness summary (skew of H_neg + raw mAP) for a given (q,g) embedding
# =========================================================================== #
def hub_and_map(qf, gf, q_pid, q_cam, g_pid, g_cam, k_main):
    qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
    gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
    sim = qf @ gf.T
    dm = 1.0 - sim
    res = eval_map(dm, q_pid, q_cam, g_pid, g_cam)
    H_neg, _ = compute_Hk(sim, q_pid, g_pid, k_main, signed='neg')
    return res, skewness(H_neg.astype(float)), float(H_neg.max()), dm, sim


# =========================================================================== #
# MAIN
# =========================================================================== #
def main():
    DS = cli.dataset
    print("#" * 84)
    print(f"# OSAC KILL-SWITCH  dataset={DS}  ckpt_dir={cli.ckpt_dir}")
    print(f"# epochs requested={cli.epochs}  final_epoch={cli.final_epoch}  smoke={cli.smoke}")
    print("#" * 84)

    want_train = not cli.no_train_feats

    # ----------------------------------------------------------------------- #
    # PASS A — per-epoch SPECTRAL TRAJECTORY (Test 1 / D4)
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 84)
    print("CORE TEST 1 / D4 — per-epoch spectral OVER-COLLAPSE trajectory")
    print("  (gallery-feature spectrum; NC1/proto on SEEN train ids)")
    print("=" * 84)
    traj = []
    final_pack = None
    for ep in cli.epochs:
        got = get_or_extract(ep, want_train)
        if got is None:
            print(f"  [ep{ep}] no checkpoint -> SKIP")
            continue
        q, g, tr, proto = got
        # drop junk gallery (market distractors pid==-1)
        keep_g = g['pid'] != -1
        for key in ('feat', 'pid', 'cam', 'name'):
            g[key] = g[key][keep_g]
        gf = g['feat'].astype(np.float64)
        gf /= (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
        qf = q['feat'].astype(np.float64)
        qf /= (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)

        eigs_g = covariance_eigvals(gf)
        er = effective_rank(eigs_g)
        sr = stable_rank(eigs_g)
        e1 = topk_energy(eigs_g, 1)
        e10 = topk_energy(eigs_g, 10)
        # sanity mAP (global single vector)
        res = eval_map(1.0 - (qf @ gf.T), q['pid'], q['cam'], g['pid'], g['cam'])
        # NC1 / proto on train (seen)
        nc1 = sw = sb = float('nan'); palign = float('nan'); ncls = 0
        if tr is not None:
            tf = tr['feat'].astype(np.float64)
            tf /= (np.linalg.norm(tf, axis=1, keepdims=True) + 1e-12)
            nc1, sw, sb = nc1_ratio(tf, tr['pid'])
            palign, ncls = proto_alignment(tf, tr['pid'], proto)
        traj.append(dict(ep=ep, er=er, sr=sr, e1=e1, e10=e10, nc1=nc1,
                         palign=palign, mAP=res['mAP'], r1=res['r1']))
        print(f"  ep{ep:3d}: eff_rank={er:7.2f}  stable_rank={sr:7.2f}  "
              f"topPC1={100*e1:5.2f}%  top10={100*e10:5.2f}%  "
              f"NC1={nc1:.4f}  proto_align={palign:.4f}  | mAP={res['mAP']:.2f} R1={res['r1']:.2f}")
        if ep == cli.final_epoch:
            final_pack = (q, g, tr, proto)

    # trajectory verdict (late-epoch deltas)
    print("\n  -- trajectory deltas (late training: does loss-still-dropping coincide with collapse?) --")
    if len(traj) >= 2:
        eps = [t['ep'] for t in traj]
        def delta(field, a, b):
            ta = next((t for t in traj if t['ep'] == a), None)
            tb = next((t for t in traj if t['ep'] == b), None)
            if ta is None or tb is None:
                return None
            return tb[field] - ta[field]
        a, b = traj[-2]['ep'], traj[-1]['ep']
        e_early = traj[0]['ep']
        for (lo, hi, tag) in [(a, b, f'{a}->{b} (last interval)'),
                              (80, 120, '80->120'),
                              (e_early, b, f'{e_early}->{b} (full)')]:
            der = delta('er', lo, hi); de1 = delta('e1', lo, hi)
            dnc = delta('nc1', lo, hi); dmap = delta('mAP', lo, hi)
            if der is None:
                continue
            print(f"    {tag:24s}: d_eff_rank={der:+7.3f}  d_topPC1={100*de1:+.3f}%  "
                  f"d_NC1={dnc:+.5f}  d_mAP={dmap:+.3f}")
        print("    OVER-COLLAPSE signature = eff_rank DOWN, topPC1 UP, NC1 DOWN while mAP flat/down.")
    else:
        print("    [WARN] <2 epochs available -> NO trajectory; Test 1/D4 reported as MISSING.")

    if final_pack is None:
        # fall back to the last available epoch for the cross-sectional tests
        print(f"\n[WARN] final_epoch {cli.final_epoch} ckpt missing; using last available for X-section.")
        last_ep = traj[-1]['ep'] if traj else None
        if last_ep is None:
            print("[ABORT] no checkpoints at all."); return
        final_pack = get_or_extract(last_ep, want_train)
        q, g, tr, proto = final_pack
        keep_g = g['pid'] != -1
        for key in ('feat', 'pid', 'cam', 'name'):
            g[key] = g[key][keep_g]
    else:
        q, g, tr, proto = final_pack

    # ----------------------------------------------------------------------- #
    # CROSS-SECTIONAL ANALYSIS at final_epoch (Test 2 / 3 + D1-D3)
    # ----------------------------------------------------------------------- #
    if cli.smoke > 0:
        for key in ('feat', 'pid', 'cam', 'name'):
            q[key] = q[key][:cli.smoke]
        print(f"\n[SMOKE] capped query -> {len(q['name'])}")

    qf = q['feat'].astype(np.float64); gf = g['feat'].astype(np.float64)
    qf /= (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
    gf /= (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
    q_pid, q_cam = q['pid'], q['cam']
    g_pid, g_cam = g['pid'], g['cam']
    Nq, Ng, D = qf.shape[0], gf.shape[0], qf.shape[1]
    sim = qf @ gf.T
    dm = 1.0 - sim
    base = eval_map(dm, q_pid, q_cam, g_pid, g_cam)
    print("\n" + "=" * 84)
    print(f"CROSS-SECTION at ep{cli.final_epoch}  Nq={Nq} Ng={Ng} D={D}")
    print(f"[SANITY] frozen cosine (global feat) mAP={base['mAP']:.2f} R1={base['r1']:.2f} "
          f"R5={base['r5']:.2f} R10={base['r10']:.2f} nq={base['nq']}")
    print("=" * 84)

    # gallery PCA (the de-collapse basis, fit unsupervised on gallery)
    g_mu, g_V, g_eigs = fit_pca(gf)
    er_g = effective_rank(g_eigs)
    print(f"[spectrum @ final] gallery eff_rank={er_g:.2f} topPC1={100*topk_energy(g_eigs,1):.2f}% "
          f"top10={100*topk_energy(g_eigs,10):.2f}%  (D={D})")

    # per-query AP + hubness at final
    aps = per_query_ap(dm, q_pid, q_cam, g_pid, g_cam)
    err = 1.0 - aps
    valid = aps >= 0
    H_neg, TK = compute_Hk(sim, q_pid, g_pid, cli.k_main, signed='neg')
    M = query_hub_mass(TK, H_neg, q_pid, g_pid)

    # query energy on gallery TOP-PCs (the collapse axes): fraction of (q-mu) norm^2
    # that lives in the top-m gallery PCs. Use top-10 as the "dominant collapse" set.
    def query_topPC_energy(X, mu, V, m):
        Xc = X - mu
        coord = Xc @ V[:, :m]                     # (n, m)
        num = (coord ** 2).sum(1)
        den = (Xc ** 2).sum(1) + 1e-12
        return num / den
    qe_top1 = query_topPC_energy(qf, g_mu, g_V, 1)
    qe_top10 = query_topPC_energy(qf, g_mu, g_V, 10)

    # prototype alignment per query: max cos to a seen prototype (over-aligned queries
    # sit closer to seen geometry -> less unseen-distinctive). proto (C,D).
    if proto is not None and proto.size:
        P = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-12)
        q_proto_max = (qf @ P.T).max(1)           # (Nq,)
        q_proto_mean = (qf @ P.T).mean(1)
    else:
        q_proto_max = np.full(Nq, np.nan)
        q_proto_mean = np.full(Nq, np.nan)

    # cheap proxies
    feat_norm_raw = np.linalg.norm(q['feat'].astype(np.float64), axis=1)  # pre-renorm norm
    sim_sorted = np.sort(sim, axis=1)
    top1margin = sim_sorted[:, -1] - sim_sorted[:, -2]
    same_cam_frac = np.zeros(Nq)
    for col in range(cli.k_main):
        gj = TK[:, col]
        same_cam_frac += (g_cam[gj] == q_cam)
    same_cam_frac /= cli.k_main

    # ======================================================================= #
    # CORE TEST 2 — collapse <-> retrieval failure
    # ======================================================================= #
    print("\n" + "=" * 84)
    print("CORE TEST 2 — per-query AP error vs collapse signatures (top-PC energy / proto-align)")
    print("=" * 84)
    rho_e1, n2 = spearman(err[valid], qe_top1[valid])
    rho_e10, _ = spearman(err[valid], qe_top10[valid])
    rho_pmax, _ = spearman(err[valid], q_proto_max[valid])
    rho_pmean, _ = spearman(err[valid], q_proto_mean[valid])
    p_e10 = perm_pvalue(err[valid], qe_top10[valid], rho_e10, n_perm=1000)
    print(f"  rho(AP-error, query-topPC1-energy)   = {rho_e1:+.4f}  (n={n2})")
    print(f"  rho(AP-error, query-top10PC-energy)  = {rho_e10:+.4f}  (perm-p {p_e10:.4f})")
    print(f"  rho(AP-error, proto-align MAX)       = {rho_pmax:+.4f}")
    print(f"  rho(AP-error, proto-align MEAN)      = {rho_pmean:+.4f}")
    print(f"  -- cheap proxies for comparison --")
    print(f"  rho(AP-error, feat-norm)             = {spearman(err[valid], feat_norm_raw[valid])[0]:+.4f}")
    print(f"  rho(AP-error, -top1-margin)          = {spearman(err[valid], -top1margin[valid])[0]:+.4f}")
    print(f"  rho(AP-error, same-cam-frac)         = {spearman(err[valid], same_cam_frac[valid])[0]:+.4f}")

    # M(q) ~ top-PC energy : hubness IS a collapse symptom
    rho_Me1, _ = spearman(M[valid], qe_top1[valid])
    rho_Me10, _ = spearman(M[valid], qe_top10[valid])
    print(f"\n  [hubness-is-symptom] rho(M(q), query-topPC1-energy)  = {rho_Me1:+.4f}")
    print(f"  [hubness-is-symptom] rho(M(q), query-top10PC-energy) = {rho_Me10:+.4f}")

    # ======================================================================= #
    # D3 — partial corr controlling camera / norm / margin
    # ======================================================================= #
    print("\n" + "=" * 84)
    print("D3 — top-PC energy <-> AP-error AFTER controlling camera/norm/margin")
    print("=" * 84)
    cov_stack = np.column_stack([feat_norm_raw[valid], top1margin[valid], same_cam_frac[valid]])
    pr_e10_norm, _ = partial_spearman(err[valid], qe_top10[valid], feat_norm_raw[valid])
    pr_e10_mar, _ = partial_spearman(err[valid], qe_top10[valid], top1margin[valid])
    pr_e10_cam, _ = partial_spearman(err[valid], qe_top10[valid], same_cam_frac[valid])
    pr_e10_all, nall = partial_spearman(err[valid], qe_top10[valid], cov_stack)
    print(f"  partial rho(AP-err, top10PC-energy | norm)         = {pr_e10_norm:+.4f}")
    print(f"  partial rho(AP-err, top10PC-energy | top1-margin)  = {pr_e10_mar:+.4f}")
    print(f"  partial rho(AP-err, top10PC-energy | same-cam)     = {pr_e10_cam:+.4f}")
    print(f"  partial rho(AP-err, top10PC-energy | norm+margin+cam) = {pr_e10_all:+.4f}  (n={nall})")
    if proto is not None and proto.size:
        pr_pmax_all, _ = partial_spearman(err[valid], q_proto_max[valid], cov_stack)
        print(f"  partial rho(AP-err, proto-align MAX | norm+margin+cam) = {pr_pmax_all:+.4f}")
    print("  D3 PASS if these stay clearly nonzero (collapse not reducible to cheap difficulty).")

    # ======================================================================= #
    # CORE TEST 3 — DE-COLLAPSE INTERVENTION (ABTT + whitening): hubness + raw mAP
    # ======================================================================= #
    print("\n" + "=" * 84)
    print("CORE TEST 3 — de-collapse (ABTT remove top-m gallery PCs / ZCA whitening)")
    print("=" * 84)
    base_res, base_skew, base_hmax, _, _ = hub_and_map(qf, gf, q_pid, q_cam, g_pid, g_cam, cli.k_main)
    print(f"  baseline                : mAP={base_res['mAP']:.3f} R1={base_res['r1']:.3f}  "
          f"H_neg skew={base_skew:.3f} max={base_hmax:.0f}")
    abtt_results = {}
    best_abtt = dict(m=0, mAP=base_res['mAP'], qf=qf, gf=gf)
    for m in cli.abtt_ms:
        which = np.arange(m)
        qf_a = abtt_remove(qf, g_mu, g_V, which)
        gf_a = abtt_remove(gf, g_mu, g_V, which)
        r, sk, hm, _, _ = hub_and_map(qf_a, gf_a, q_pid, q_cam, g_pid, g_cam, cli.k_main)
        abtt_results[m] = dict(mAP=r['mAP'], r1=r['r1'], skew=sk, hmax=hm, qf=qf_a, gf=gf_a)
        flag = ''
        if r['mAP'] > best_abtt['mAP']:
            best_abtt = dict(m=m, mAP=r['mAP'], r1=r['r1'], qf=qf_a, gf=gf_a); flag = '  <== best'
        print(f"  ABTT top-{m:<2d} removed   : mAP={r['mAP']:.3f} (d{r['mAP']-base_res['mAP']:+.3f}) "
              f"R1={r['r1']:.3f}  H_neg skew={sk:.3f} (d{sk-base_skew:+.3f}) max={hm:.0f}{flag}")
    # whitening sweep (n_keep)
    print("  -- ZCA whitening (gallery-fit), vary #components kept --")
    best_white = dict(nk=0, mAP=base_res['mAP'], qf=qf, gf=gf)
    for nk in [64, 128, 256, 512, min(D, 1024)]:
        if nk > D:
            continue
        qf_w, gf_w = zca_whiten(qf, gf, g_mu, g_V, g_eigs, eps=1e-3, n_keep=nk)
        r, sk, hm, _, _ = hub_and_map(qf_w, gf_w, q_pid, q_cam, g_pid, g_cam, cli.k_main)
        flag = ''
        if r['mAP'] > best_white['mAP']:
            best_white = dict(nk=nk, mAP=r['mAP'], r1=r['r1'], qf=qf_w, gf=gf_w); flag = '  <== best'
        print(f"  whiten keep-{nk:<4d}      : mAP={r['mAP']:.3f} (d{r['mAP']-base_res['mAP']:+.3f}) "
              f"R1={r['r1']:.3f}  H_neg skew={sk:.3f} max={hm:.0f}{flag}")
    print(f"\n  >> best ABTT: top-{best_abtt['m']} removed -> mAP={best_abtt['mAP']:.3f} "
          f"(d{best_abtt['mAP']-base_res['mAP']:+.3f})")
    print(f"  >> best whiten: keep-{best_white['nk']} -> mAP={best_white['mAP']:.3f} "
          f"(d{best_white['mAP']-base_res['mAP']:+.3f})")

    # gain concentration: does ABTT gain concentrate on high-M(q) (hub) queries?
    if best_abtt['m'] > 0:
        aps_a = per_query_ap(1.0 - best_abtt['qf'] @ best_abtt['gf'].T,
                             q_pid, q_cam, g_pid, g_cam)
        dgain = aps_a - aps
        sel = (aps >= 0) & (aps_a >= 0)
        rho_gain, _ = spearman(M[sel], dgain[sel])
        rho_gain_e, _ = spearman(qe_top10[sel], dgain[sel])
        print(f"  gain concentration: rho(M(q), ABTT AP-gain)={rho_gain:+.4f}  "
              f"rho(top10PC-energy, gain)={rho_gain_e:+.4f}  (expect POSITIVE)")

    # ======================================================================= #
    # DESTRUCTIVE CONTROLS
    # ======================================================================= #
    print("\n" + "#" * 84)
    print("DESTRUCTIVE CONTROLS (decide life/death)")
    print("#" * 84)

    # ---- D1: random-PC / bottom-PC removal must be WORSE than top-PC ----
    print("\n-- D1: random-PC & bottom-PC removal vs TOP-PC removal --")
    m_star = best_abtt['m'] if best_abtt['m'] > 0 else cli.abtt_ms[len(cli.abtt_ms)//2]
    # top-m (already): use best m
    qf_top = abtt_remove(qf, g_mu, g_V, np.arange(m_star))
    gf_top = abtt_remove(gf, g_mu, g_V, np.arange(m_star))
    r_top = eval_map(1.0 - qf_top @ gf_top.T, q_pid, q_cam, g_pid, g_cam)
    # bottom-m (lowest-energy PCs)
    which_bot = np.arange(D - m_star, D)
    qf_bot = abtt_remove(qf, g_mu, g_V, which_bot)
    gf_bot = abtt_remove(gf, g_mu, g_V, which_bot)
    r_bot = eval_map(1.0 - qf_bot @ gf_bot.T, q_pid, q_cam, g_pid, g_cam)
    # random-m PCs (avg over a few draws)
    rand_maps = []
    for _ in range(5):
        which_rnd = RNG.choice(D, m_star, replace=False)
        qf_r = abtt_remove(qf, g_mu, g_V, which_rnd)
        gf_r = abtt_remove(gf, g_mu, g_V, which_rnd)
        rand_maps.append(eval_map(1.0 - qf_r @ gf_r.T, q_pid, q_cam, g_pid, g_cam)['mAP'])
    print(f"  m={m_star}: TOP-PC removal   mAP={r_top['mAP']:.3f} (d{r_top['mAP']-base['mAP']:+.3f})")
    print(f"  m={m_star}: BOTTOM-PC removal mAP={r_bot['mAP']:.3f} (d{r_bot['mAP']-base['mAP']:+.3f})")
    print(f"  m={m_star}: RANDOM-PC removal mAP={np.mean(rand_maps):.3f}+-{np.std(rand_maps):.3f} "
          f"(d{np.mean(rand_maps)-base['mAP']:+.3f})")
    print(f"  >> D1 PASS if TOP-PC removal >> bottom/random (collapse axes are special).")

    # ---- D2 (MOST CRITICAL): de-collapse vs k-reciprocal residual ----
    print("\n" + "=" * 84)
    print("D2 (CRITICAL) — DE-COLLAPSE residual AFTER k-reciprocal re-ranking")
    print("=" * 84)
    # baseline + k-reciprocal
    dm_rr_base = kreciprocal_rerank(qf, gf, q_cam, g_cam, k1=20, k2=6, lam=0.3)
    r_rr_base = eval_map(dm_rr_base, q_pid, q_cam, g_pid, g_cam)
    # ABTT(best) + k-reciprocal
    qf_b, gf_b = best_abtt['qf'], best_abtt['gf']
    dm_rr_abtt = kreciprocal_rerank(qf_b, gf_b, q_cam, g_cam, k1=20, k2=6, lam=0.3)
    r_rr_abtt = eval_map(dm_rr_abtt, q_pid, q_cam, g_pid, g_cam)
    # whitening(best) + k-reciprocal
    qf_w, gf_w = best_white['qf'], best_white['gf']
    dm_rr_white = kreciprocal_rerank(qf_w, gf_w, q_cam, g_cam, k1=20, k2=6, lam=0.3)
    r_rr_white = eval_map(dm_rr_white, q_pid, q_cam, g_pid, g_cam)

    print(f"  baseline (raw)                 mAP={base['mAP']:.3f}  R1={base['r1']:.3f}")
    print(f"  baseline + k-reciprocal        mAP={r_rr_base['mAP']:.3f}  R1={r_rr_base['r1']:.3f}  "
          f"(RR lift d{r_rr_base['mAP']-base['mAP']:+.3f})")
    print(f"  ABTT(top-{best_abtt['m']}) raw           mAP={best_abtt['mAP']:.3f}  "
          f"(raw lift d{best_abtt['mAP']-base['mAP']:+.3f})")
    print(f"  ABTT(top-{best_abtt['m']}) + k-reciprocal mAP={r_rr_abtt['mAP']:.3f}  R1={r_rr_abtt['r1']:.3f}")
    print(f"  whiten(keep-{best_white['nk']}) + k-recip   mAP={r_rr_white['mAP']:.3f}  R1={r_rr_white['r1']:.3f}")
    resid_abtt = r_rr_abtt['mAP'] - r_rr_base['mAP']
    resid_white = r_rr_white['mAP'] - r_rr_base['mAP']
    print(f"\n  >> RESIDUAL (ABTT+RR) - (baseline+RR)   = {resid_abtt:+.4f}")
    print(f"  >> RESIDUAL (whiten+RR) - (baseline+RR) = {resid_white:+.4f}")
    print(f"  >> D2 VERDICT: OSAC survives ONLY if de-collapse keeps a CLEAR POSITIVE residual")
    print(f"     AFTER k-reciprocal. If residual ~0 or negative -> re-ranking already")
    print(f"     recovers everything -> OSAC DIES (same death as Hubness).")

    # ======================================================================= #
    # FINAL SUMMARY
    # ======================================================================= #
    print("\n" + "#" * 84)
    print(f"SUMMARY / VERDICT  ({DS})")
    print("#" * 84)
    if len(traj) >= 2:
        t0, tN = traj[0], traj[-1]
        print(f"  TRAJECTORY ep{t0['ep']}->ep{tN['ep']}: "
              f"eff_rank {t0['er']:.1f}->{tN['er']:.1f} ({tN['er']-t0['er']:+.2f}), "
              f"topPC1 {100*t0['e1']:.2f}%->{100*tN['e1']:.2f}% ({100*(tN['e1']-t0['e1']):+.2f}), "
              f"NC1 {t0['nc1']:.4f}->{tN['nc1']:.4f}, mAP {t0['mAP']:.1f}->{tN['mAP']:.1f}")
        # late interval specifically
        if len(traj) >= 2:
            la, lb = traj[-2], traj[-1]
            print(f"  LATE interval ep{la['ep']}->ep{lb['ep']}: "
                  f"d_eff_rank={lb['er']-la['er']:+.3f}  d_topPC1={100*(lb['e1']-la['e1']):+.3f}%  "
                  f"d_NC1={lb['nc1']-la['nc1']:+.5f}  d_mAP={lb['mAP']-la['mAP']:+.3f}")
    else:
        print(f"  TRAJECTORY: MISSING (<2 epoch ckpts).")
    print(f"  T2 rho(AP-err, top10PC-energy)    = {rho_e10:+.4f}  (perm-p {p_e10:.4f})")
    print(f"  T2 rho(AP-err, proto-align max)   = {rho_pmax:+.4f}")
    print(f"  T2 rho(M(q), top10PC-energy)      = {rho_Me10:+.4f}  [hubness=symptom]")
    print(f"  D3 partial(AP-err,topPC | n+m+c)  = {pr_e10_all:+.4f}  (want clearly !=0)")
    print(f"  T3 best ABTT raw gain             = {best_abtt['mAP']-base['mAP']:+.3f} (top-{best_abtt['m']})")
    print(f"  T3 best whiten raw gain           = {best_white['mAP']-base['mAP']:+.3f} (keep-{best_white['nk']})")
    print(f"  D1 top vs bottom vs random        = {r_top['mAP']-base['mAP']:+.3f} / "
          f"{r_bot['mAP']-base['mAP']:+.3f} / {np.mean(rand_maps)-base['mAP']:+.3f}")
    print(f"  D2 RESIDUAL after k-recip (ABTT)  = {resid_abtt:+.4f}   (whiten {resid_white:+.4f})")
    print(f"      [baseline+RR={r_rr_base['mAP']:.3f}  ABTT+RR={r_rr_abtt['mAP']:.3f}]")
    print("\n[done] OSAC kill-switch complete.")

    # dump machine-readable summary
    out = dict(dataset=DS, trajectory=traj,
               final_epoch=cli.final_epoch, base_mAP=base['mAP'],
               T2_rho_err_top10=rho_e10, T2_perm_p=p_e10, T2_rho_err_protomax=rho_pmax,
               T2_rho_M_top10=rho_Me10,
               D3_partial_all=pr_e10_all,
               T3_best_abtt_m=best_abtt['m'], T3_best_abtt_gain=best_abtt['mAP']-base['mAP'],
               T3_best_white_nk=best_white['nk'], T3_best_white_gain=best_white['mAP']-base['mAP'],
               D1_top=r_top['mAP']-base['mAP'], D1_bottom=r_bot['mAP']-base['mAP'],
               D1_random=float(np.mean(rand_maps))-base['mAP'],
               D2_baseRR=r_rr_base['mAP'], D2_abttRR=r_rr_abtt['mAP'], D2_whiteRR=r_rr_white['mAP'],
               D2_residual_abtt=resid_abtt, D2_residual_white=resid_white)
    jpath = os.path.join(cli.cache_dir, f'osac_summary_{DS}.json')
    with open(jpath, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[json] summary -> {jpath}")


if __name__ == '__main__':
    main()
