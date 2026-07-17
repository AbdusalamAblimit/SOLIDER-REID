#!/usr/bin/env python3
"""GOPL — Geometry-Ordered Positive Learning — ZERO-TRAINING kill-switch.

Re-frame under test (GOPL_KILLSWITCH_DESIGN.md):
    Occluded ReID is NOT "feature alignment is too weak" but "the GRANULARITY of
    the positive relation is wrong": supervised ReID pulls ALL same-ID pairs
    together as EQUIVALENT positives, yet two images of one identity may share
    almost no co-observable body surface (one occludes legs, the other the
    torso -> only the head is common). Under disjoint visible surfaces the
    "same identity" label is OVER-STRONG supervision and pollutes the manifold.

    GOPL: SMPL co-visible-surface overlap is used ONLY as a *reliability meter*
    on same-ID positive edges -- NOT identity rep, NOT inference matching, NOT
    part alignment, NOT augmentation.

This script trains NOTHING. It freezes a strong occluded_duke ReID ckpt
(exp255: PSG + LGPA-D + GCN512 2-stage, final mAP 72.7/82.5), extracts a single
clean GLOBAL feature per image (POSE_TEST_FEAT='global' -> the model returns only
its global branch vector; PSG still gates the backbone, so the feature is the real
trained eval feature), aligns features to the SMPL cache BY IMAGE NAME, and runs
the design's core tests 1-4 + destructive controls D1-D5 for two co-visibility
metrics:
    cov2d  -- 2D-joint co-visibility IoU  (VPM/QPM-style 2D-visibility control)
    cov3d  -- 3D-joint co-visibility IoU with self-occlusion (the GOPL claim)
D4 (cov3d must clearly beat cov2d) is the make-or-break line.

Run on lab-3090-d:
    cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 python3 \
      experiments/cargo_cvpb/cvpb_gopl_killswitch.py \
      --config configs/occluded_duke/pose_psg_lgpa_gcn512_2stage_small.yml \
      --ckpt log/occluded_duke/exp255_small_gcn512_2stage/transformer_120.pth \
      --smpl_dir cache/smpl_geom 2>&1 | tee /tmp/cvpb_gopl.log
    # smoke first: add  --smoke 200
"""
import os, sys, time, argparse, json
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
# repo root = three levels up from experiments/cargo_cvpb/
_repo = os.path.abspath(os.path.join(_here, '..', '..'))
sys.path.insert(0, _repo)

ap = argparse.ArgumentParser()
ap.add_argument('--config', default='configs/occluded_duke/pose_psg_lgpa_gcn512_2stage_small.yml')
ap.add_argument('--ckpt', default='log/occluded_duke/exp255_small_gcn512_2stage/transformer_120.pth')
ap.add_argument('--smpl_dir', default='cache/smpl_geom')
ap.add_argument('--cache_feat', default='/tmp/gopl_feats.npz',
                help='where to dump/reuse extracted features (skip extraction if exists)')
ap.add_argument('--reuse_feat', action='store_true', help='reuse --cache_feat if present')
ap.add_argument('--smoke', type=int, default=0, help='if >0 cap #query for a fast smoke run')
ap.add_argument('--conf_thr', type=float, default=0.0,
                help='SMPL per-IMAGE conf gate (scalar in this cache); images below are still '
                     'usable but flagged. Joint visibility comes from 2D in-bounds, NOT conf.')
ap.add_argument('--margin', type=float, default=0.0,
                help='in-bounds margin (px) when testing pj2d inside [0,W]x[0,H]')
ap.add_argument('--seed', type=int, default=42)
cli = ap.parse_args()
np.random.seed(cli.seed)
RNG = np.random.RandomState(cli.seed)


# =========================================================================== #
# 1. FEATURE EXTRACTION  (frozen exp255, POSE_TEST_FEAT='global', by img name)
# =========================================================================== #
def extract_features():
    import torch
    import torch.nn.functional as F
    from config import cfg
    from datasets import make_dataloader
    from model import make_model

    cfg.merge_from_file(os.path.join(_repo, cli.config))
    # force a clean single-vector global feature; frozen; standard cosine eval
    cfg.merge_from_list([
        'TEST.WEIGHT', os.path.join(_repo, cli.ckpt),
        'MODEL.POSE_TEST_FEAT', 'global',
        'TEST.NECK_FEAT', 'after',
        'TEST.FEAT_NORM', 'yes',
        'TEST.IMS_PER_BATCH', 64,
    ])
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
            # pose_val_collate_fn -> (imgs, pids, camids, camids_tensor, viewids, img_paths, pose_dict)
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
            feat = F.normalize(feat, p=2, dim=1)   # cosine-eval normalization
            feats.append(feat.cpu().numpy().astype(np.float32))
            pids.extend([int(x) for x in b_pids])
            camids.extend([int(x) for x in (b_camids_t.tolist())])
            names.extend([os.path.basename(p) for p in img_paths])
            if bi % 20 == 0:
                print(f"  [extract] batch {bi}/{len(val_loader)} ({time.time()-t0:.0f}s)", flush=True)

    feats = np.concatenate(feats, 0)
    pids = np.asarray(pids); camids = np.asarray(camids); names = np.asarray(names)
    # split query / gallery: first num_query rows are query (repo convention)
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
# 2. EVAL (occluded_duke market-style: drop same pid&cam junk)
# =========================================================================== #
def cosine_distmat(qf, gf):
    return 1.0 - (qf @ gf.T)


def eval_map(distmat, q_pid, q_cam, g_pid, g_cam, max_rank=10):
    num_q, num_g = distmat.shape
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
                r5=float(all_cmc[4]) * 100, nq=nvalid)


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
# 3. SMPL CO-VISIBILITY METRICS
# =========================================================================== #
# pj2d: (N,71,2) in INPUT-image pixel coords (the cache stores ROMP 2D joints in the
#       model input space, ~ W=128 x H=384 here, with out-of-frame joints having
#       coords outside [0,W]x[0,H]). j3d: (N,71,3) normalized body-centered 3D.
# conf: (N,) per-IMAGE scalar (NOT per-joint). valid: (N,) 1 if SMPL detected.

def load_smpl(smpl_dir):
    out = {}
    for split in ('query', 'gallery', 'train'):
        p = os.path.join(_repo, smpl_dir, f'{split}.npz')
        d = np.load(p, allow_pickle=True)
        out[split] = dict(names=np.asarray(d['names']),
                          pj2d=d['pj2d'].astype(np.float64),
                          j3d=d['j3d'].astype(np.float64),
                          conf=d['conf'].astype(np.float64),
                          valid=d['valid'].astype(np.int64))
    return out


def vis2d(pj2d, valid, W, H, margin=0.0):
    """Per-joint 2D visibility (N,71): joint is 'visible' iff its projected coord lies
    inside the image frame AND the SMPL fit is valid for that image. This is exactly
    the VPM/QPM-style 2D part-visibility signal.

    pj2d is ROMP `pj2d_org` -> ORIGINAL per-image frame, so W,H are per-image arrays
    (shape (N,)) of each crop's real width/height, NOT a global model-input size."""
    x = pj2d[..., 0]; y = pj2d[..., 1]
    W = np.asarray(W, float)[:, None]; H = np.asarray(H, float)[:, None]
    inb = (x >= -margin) & (x <= W + margin) & (y >= -margin) & (y <= H + margin)
    v = inb & (valid[:, None] == 1)
    return v.astype(bool)


def _body_frame(j3d):
    """Estimate a per-image body frame from 3D joints, returning the camera-facing
    normal n_hat (N,3) (unit vector pointing toward the camera-visible/front side).

    Uses SMPL/ROMP-style joints: indices 0..23 are SMPL body joints. We use
    L/R shoulders and L/R hips to build:
      right_dir = (Rsh - Lsh) + (Rhip - Lhip)      (body left->right axis)
      up_dir    = 0.5*(Lsh+Rsh) - 0.5*(Lhip+Rhip)  (hip->shoulder = up)
      facing    = up_dir x right_dir               (frontal normal)
    The SIGN of the camera axis is resolved data-adaptively in cov3d via calibration
    against 2D in-bounds (the front side should be the more 2D-visible side)."""
    # SMPL joint order (ROMP/SMPL 24): 16=L_shoulder 17=R_shoulder 1=L_hip 2=R_hip
    LSH, RSH, LHIP, RHIP = 16, 17, 1, 2
    Lsh = j3d[:, LSH]; Rsh = j3d[:, RSH]; Lhip = j3d[:, LHIP]; Rhip = j3d[:, RHIP]
    right_dir = (Rsh - Lsh) + (Rhip - Lhip)               # (N,3)
    up_dir = 0.5 * (Lsh + Rsh) - 0.5 * (Lhip + Rhip)      # (N,3)
    facing = np.cross(up_dir, right_dir)                  # (N,3) frontal normal
    nrm = np.linalg.norm(facing, axis=1, keepdims=True)
    facing = facing / np.maximum(nrm, 1e-8)
    center = 0.25 * (Lsh + Rsh + Lhip + Rhip)            # torso center
    return facing, center, right_dir, up_dir


def vis3d(j3d, valid, pj2d, W, H, tau=0.02, return_pure=False):
    """Per-joint 3D self-occlusion visibility (N,71).

    A joint is camera-FACING iff the body surface at that joint faces the camera.
    Joint-level approximation (no mesh): project each joint's offset from the torso
    center onto the frontal normal n_hat; joints on the camera-facing half-space face
    the camera, joints on the far (back) half-space are self-occluded.

      depth_j = (joint - center) . n_hat
      front   = (depth_j * S) >= -tau          (S = single GLOBAL camera-axis sign)

    Sign ambiguity: normalized j3d has no absolute camera axis, so the sign S that
    makes 'front' joints land MORE inside the 2D frame is chosen ONCE GLOBALLY (a
    single bit over the whole split), not per image -- this avoids leaking 2D info
    into the 3D signal per-pair.

    The reported 3D visibility ANDs front-facing with 2D in-frame:
        vis3d = front-facing  AND  in-2D-frame
    i.e. it models BOTH occlusion sources (self-occlusion removes back joints that 2D
    visibility wrongly keeps; cropping removes out-of-frame joints). cov3d is thus a
    strict refinement of cov2d. `return_pure=True` also returns the front-facing-ONLY
    mask (independent of 2D) for a non-circular D4 check."""
    N = j3d.shape[0]
    facing, center, _, _ = _body_frame(j3d)               # (N,3),(N,3)
    off = j3d - center[:, None, :]                        # (N,71,3) offsets
    depth = np.einsum('njd,nd->nj', off, facing)          # (N,71) signed depth along normal
    inb = vis2d(pj2d, valid, W, H)                        # (N,71) 2D in-frame (per-image W,H)
    vmask = (valid == 1)
    # choose ONE global sign: front (depth*S>=-tau) should overlap 2D in-frame more.
    agree_pos = (((depth >= -tau) == inb) & vmask[:, None]).sum()
    agree_neg = (((-depth >= -tau) == inb) & vmask[:, None]).sum()
    S = 1 if agree_pos >= agree_neg else -1
    front = (depth * S) >= -tau                           # (N,71) front-facing
    front &= vmask[:, None]
    out = front & inb                                     # 3D self-occ AND 2D in-frame
    if return_pure:
        return out, front, np.full(N, S, dtype=np.int64)
    return out, np.full(N, S, dtype=np.int64)


def cov_iou(vis_a, vis_b):
    """Co-visibility IoU between two per-joint boolean visibility vectors."""
    inter = np.logical_and(vis_a, vis_b).sum()
    union = np.logical_or(vis_a, vis_b).sum()
    return inter / union if union > 0 else 0.0


# =========================================================================== #
# helpers: stats
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


def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling for z: correlate the rank-
    residuals of x|z and y|z (linear regression on ranks)."""
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    if len(x) < 5:
        return float('nan'), 0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rz = np.argsort(np.argsort(z)).astype(float)
    Z = np.column_stack([np.ones_like(rz), rz])
    def resid(r):
        beta, *_ = np.linalg.lstsq(Z, r, rcond=None)
        return r - Z @ beta
    ex, ey = resid(rx), resid(ry)
    denom = np.sqrt((ex**2).sum() * (ey**2).sum())
    rho = float((ex * ey).sum() / denom) if denom > 0 else float('nan')
    return rho, len(x)


def perm_pvalue(x, y, rho_obs, n_perm=1000):
    """Two-sided permutation p-value for a Spearman rho (shuffle y)."""
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


def quartile_buckets(values):
    """Rank-based quartiles: split sorted order into 4 equal groups. Robust to heavy
    saturation/ties (np.quantile-based digitize leaves empty buckets when a value like
    1.0 sits on the top edge). Returns idx in 0..3 (Q0=lowest)."""
    values = np.asarray(values, float)
    n = len(values)
    order = np.argsort(values, kind='stable')
    idx = np.empty(n, dtype=np.int64)
    splits = np.array_split(order, 4)
    for b, s in enumerate(splits):
        idx[s] = b
    qs = np.quantile(values, [0.25, 0.5, 0.75])
    return idx, qs, None


# =========================================================================== #
# MAIN
# =========================================================================== #
def main():
    # ---- features ----
    if cli.reuse_feat and os.path.exists(cli.cache_feat):
        z = np.load(cli.cache_feat, allow_pickle=True)
        q = dict(feat=z['q_feat'], pid=z['q_pid'], cam=z['q_cam'], name=z['q_name'])
        g = dict(feat=z['g_feat'], pid=z['g_pid'], cam=z['g_cam'], name=z['g_name'])
        print(f"[reuse] features from {cli.cache_feat}: q={len(q['name'])} g={len(g['name'])}")
    else:
        q, g = extract_features()

    # ---- sanity: frozen cosine mAP ----
    qf = q['feat'].astype(np.float64); gf = g['feat'].astype(np.float64)
    qf /= (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
    gf /= (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
    dm = cosine_distmat(qf, gf)
    res = eval_map(dm, q['pid'], q['cam'], g['pid'], g['cam'])
    print(f"\n[SANITY] frozen cosine (global feat) mAP={res['mAP']:.2f} R1={res['r1']:.2f} "
          f"R5={res['r5']:.2f} nq={res['nq']}  (exp255 fused-feat ref 72.7/82.5; "
          f"this is the GLOBAL-branch single-vector, expected somewhat lower)")

    # ---- SMPL: align by image name ----
    smpl = load_smpl(cli.smpl_dir)
    # pj2d is ROMP pj2d_org -> ORIGINAL per-image crop frame. Need each image's real
    # (W,H) for the in-bounds visibility test (Occluded-Duke crops vary, ~128x256).
    qdir = os.path.join(_repo, 'data/occluded_duke/query')
    gdir = os.path.join(_repo, 'data/occluded_duke/bounding_box_test')
    print(f"\n[SMPL] per-image original-frame in-bounds test; margin={cli.margin}")

    def _img_wh(split_dir, names):
        from PIL import Image
        Ws = np.zeros(len(names)); Hs = np.zeros(len(names))
        for i, nm in enumerate(names):
            try:
                w, h = Image.open(os.path.join(split_dir, nm)).size
            except Exception:
                w, h = 128, 256
            Ws[i] = w; Hs[i] = h
        return Ws, Hs

    def build_vis(split_dir, smpl_split):
        names = list(smpl_split['names'])
        name2i = {n: i for i, n in enumerate(names)}
        Ws, Hs = _img_wh(split_dir, names)
        v2 = vis2d(smpl_split['pj2d'], smpl_split['valid'], Ws, Hs, cli.margin)
        v3, v3pure, signs = vis3d(smpl_split['j3d'], smpl_split['valid'],
                                  smpl_split['pj2d'], Ws, Hs, return_pure=True)
        return name2i, v2, v3, v3pure, smpl_split['valid'], smpl_split['conf'], signs

    qn2i, qv2, qv3, qv3p, qvalid, qconf, qsign = build_vis(qdir, smpl['query'])
    gn2i, gv2, gv3, gv3p, gvalid, gconf, gsign = build_vis(gdir, smpl['gallery'])

    # diagnostics on visibility
    print(f"[SMPL] query  valid={int(qvalid.sum())}/{len(qvalid)}  "
          f"mean #vis2d={qv2.sum(1)[qvalid==1].mean():.1f}  "
          f"mean #vis3d={qv3.sum(1)[qvalid==1].mean():.1f}  "
          f"3D sign +/- = {int((qsign>0).sum())}/{int((qsign<0).sum())}")
    print(f"[SMPL] gallery valid={int(gvalid.sum())}/{len(gvalid)}  "
          f"mean #vis2d={gv2.sum(1)[gvalid==1].mean():.1f}  "
          f"mean #vis3d={gv3.sum(1)[gvalid==1].mean():.1f}")
    # how different are vis2d and vis3d? (per-image jaccard)
    both = (qvalid == 1)
    jac = []
    for i in np.where(both)[0]:
        u = np.logical_or(qv2[i], qv3[i]).sum()
        if u > 0:
            jac.append(np.logical_and(qv2[i], qv3[i]).sum() / u)
    print(f"[SMPL] per-image Jaccard(vis2d,vis3d) on query = {np.mean(jac):.3f} "
          f"(if ~1.0, 3D self-occlusion adds nothing -> D4 will fail)")

    # ---- build the set of valid SAME-ID query<->gallery pairs (cross-cam) ----
    # We analyze same-ID pairs between query (occluded) and gallery, dropping same-cam
    # (the eval protocol's junk rule). Each pair carries: cosine dist, cov2d, cov3d,
    # #vis (occlusion proxy), bbox-area ratio (D2), camera-pair id (D2).
    print("\n[PAIRS] building same-ID query<->gallery cross-cam pairs ...")
    q_area = _areas([os.path.join(qdir, n) for n in q['name']])
    g_area = _areas([os.path.join(gdir, n) for n in g['name']])

    pairs = []   # dicts
    gpid = g['pid']; gcam = g['cam']
    # precompute per-gallery smpl indices
    g_si = np.array([gn2i.get(n, -1) for n in g['name']])
    q_si = np.array([qn2i.get(n, -1) for n in q['name']])
    cos_full = 1.0 - (qf @ gf.T)   # (Nq,Ng) cosine distance
    for i in range(len(q['name'])):
        si_q = q_si[i]
        if si_q < 0 or qvalid[si_q] != 1:
            continue
        same = np.where((gpid == q['pid'][i]) & (gcam != q['cam'][i]))[0]
        for j in same:
            sj = g_si[j]
            if sj < 0 or gvalid[sj] != 1:
                continue
            c2 = cov_iou(qv2[si_q], gv2[sj])
            c3 = cov_iou(qv3[si_q], gv3[sj])
            c3p = cov_iou(qv3p[si_q], gv3p[sj])
            nvis_min = min(int(qv2[si_q].sum()), int(gv2[sj].sum()))
            pairs.append(dict(
                qi=i, gj=j, cos=float(cos_full[i, j]),
                cov2d=c2, cov3d=c3, cov3d_pure=c3p,
                nvis=nvis_min,
                conf=float(min(qconf[si_q], gconf[sj])),
                area_ratio=_ratio(q_area[i], g_area[j]),
                campair=q['cam'][i] * 100 + gcam[j],
            ))
    P = pairs
    n = len(P)
    print(f"[PAIRS] valid same-ID cross-cam pairs with SMPL on both sides: n={n}")
    if n < 50:
        print("[ABORT] too few valid pairs; SMPL coverage on same-ID pairs is insufficient.")
        return
    cos = np.array([p['cos'] for p in P])
    c2 = np.array([p['cov2d'] for p in P])
    c3 = np.array([p['cov3d'] for p in P])
    c3p = np.array([p['cov3d_pure'] for p in P])
    nvis = np.array([p['nvis'] for p in P], float)
    arearatio = np.array([p['area_ratio'] for p in P], float)
    campair = np.array([p['campair'] for p in P])

    print(f"[PAIRS] cov2d: mean={c2.mean():.3f} sd={c2.std():.3f}  "
          f"cov3d: mean={c3.mean():.3f} sd={c3.std():.3f}  "
          f"cos-dist: mean={cos.mean():.3f} sd={cos.std():.3f}")

    # =================================================================== #
    # CORE TEST 1 — same-ID cosine distance vs (1 - cov): Spearman
    #   expect POSITIVE rho(cos, 1-cov)  == NEGATIVE rho(cos, cov)
    # =================================================================== #
    print("\n" + "=" * 78)
    print("CORE TEST 1 — same-ID cosine distance vs co-visibility (Spearman)")
    print("=" * 78)
    for tag, cov in (('cov2d', c2), ('cov3d', c3)):
        rho_cov, nn = spearman(cos, cov)         # expect negative
        rho_1mcov, _ = spearman(cos, 1.0 - cov)  # expect positive
        p = perm_pvalue(cos, cov, rho_cov, n_perm=1000)
        # control: partial out #visible joints (occlusion severity)
        rho_pj, _ = partial_spearman(cos, cov, nvis)
        print(f"  [{tag}] rho(cos, cov)={rho_cov:+.4f}  rho(cos,1-cov)={rho_1mcov:+.4f}  "
              f"perm-p={p:.4f}  n={nn}")
        print(f"         partial rho(cos, cov | #vis)={rho_pj:+.4f}  "
              f"(expect still negative if cov adds beyond raw occlusion severity)")

    # --- DECISIVE BASELINE: is cov just an occlusion-COUNT (#visible joints) proxy? ---
    #   If rho(cos, #vis) ~ rho(cos, cov) and cov dies when #vis is controlled, then
    #   "co-visibility geometry" reduces to "how occluded the harder image is".
    print("\n  -- occlusion-count baseline (the thing GOPL must beat) --")
    rho_nvis, _ = spearman(cos, -nvis)   # fewer visible joints -> larger dist (expect +? use -nvis to match 1-cov sign)
    rho_nvis_raw, _ = spearman(cos, nvis)
    print(f"     rho(cos, #vis-joints)        = {rho_nvis_raw:+.4f}  (more visible -> smaller dist if negative)")
    print(f"     rho(cos, cov2d)={spearman(cos,c2)[0]:+.4f}  rho(cos, cov3d)={spearman(cos,c3)[0]:+.4f}")
    print(f"     => cov2d signal AFTER removing #vis: {partial_spearman(cos,c2,nvis)[0]:+.4f} "
          f"(if ~0, cov2d == occlusion count, NOT co-visibility)")
    print(f"     => cov3d signal AFTER removing #vis: {partial_spearman(cos,c3,nvis)[0]:+.4f}")

    # =================================================================== #
    # CORE TEST 2 — same-ID pairs bucketed by cov quartile: mean cosine dist
    #   expect LOW-cov bucket -> HIGHER cosine distance
    # =================================================================== #
    print("\n" + "=" * 78)
    print("CORE TEST 2 — mean cosine distance by co-visibility quartile")
    print("=" * 78)
    for tag, cov in (('cov2d', c2), ('cov3d', c3)):
        idx, qs, _ = quartile_buckets(cov)
        print(f"  [{tag}] quartile edges={np.round(qs,3)}")
        means = []
        for b in range(4):
            sel = idx == b
            if sel.sum() == 0:
                means.append(np.nan); continue
            means.append(cos[sel].mean())
            print(f"     Q{b} cov in bucket (n={sel.sum():5d}): "
                  f"mean cov={cov[sel].mean():.3f}  mean cos-dist={cos[sel].mean():.4f}")
        gap = means[0] - means[3]
        print(f"     -> bottom(Q0) - top(Q3) cosine-dist gap = {gap:+.4f} "
              f"(expect POSITIVE: low overlap = larger distance)")

    # =================================================================== #
    # CORE TEST 3 — per-query AP bucketed by the true-match MAX cov
    #   expect bottom-cov queries -> lower AP/mAP
    # =================================================================== #
    print("\n" + "=" * 78)
    print("CORE TEST 3 — per-query AP by (max co-visibility over true matches)")
    print("=" * 78)
    aps = per_query_ap(dm, q['pid'], q['cam'], g['pid'], g['cam'])
    # for each query: the max cov over its valid same-ID gallery matches
    q_maxcov2 = np.full(len(q['name']), np.nan)
    q_maxcov3 = np.full(len(q['name']), np.nan)
    from collections import defaultdict
    byq2 = defaultdict(list); byq3 = defaultdict(list)
    for p in P:
        byq2[p['qi']].append(p['cov2d']); byq3[p['qi']].append(p['cov3d'])
    for i in byq2:
        q_maxcov2[i] = max(byq2[i]); q_maxcov3[i] = max(byq3[i])
    for tag, mc in (('cov2d', q_maxcov2), ('cov3d', q_maxcov3)):
        sel = (aps >= 0) & np.isfinite(mc)
        if sel.sum() < 8:
            print(f"  [{tag}] too few queries ({sel.sum()})"); continue
        idx, qs, _ = quartile_buckets(mc[sel])
        ap_sel = aps[sel]
        print(f"  [{tag}] max-cov quartile edges={np.round(qs,3)}  (n={sel.sum()})")
        bmeans = []
        for b in range(4):
            s = idx == b
            if s.sum() == 0:
                bmeans.append(np.nan); continue
            bmeans.append(100 * ap_sel[s].mean())
            print(f"     Q{b} (n={s.sum():4d}): mAP={100*ap_sel[s].mean():.2f}")
        print(f"     -> top(Q3) - bottom(Q0) mAP = {bmeans[3]-bmeans[0]:+.2f} "
              f"(expect POSITIVE: low max-overlap queries are harder)")
        rho_q, _ = spearman(mc[sel], ap_sel)
        print(f"     -> Spearman(max-cov, AP) = {rho_q:+.4f} (expect positive)")

    # =================================================================== #
    # CORE TEST 4 — hardest same-ID positives (top-10% cosine dist):
    #   fraction in the LOW-cov half should be high
    # =================================================================== #
    print("\n" + "=" * 78)
    print("CORE TEST 4 — low-cov fraction among hardest (top-10% cos-dist) same-ID pairs")
    print("=" * 78)
    k = max(1, int(0.10 * n))
    hard = np.argsort(-cos)[:k]   # largest cosine distance = hardest positives
    for tag, cov in (('cov2d', c2), ('cov3d', c3)):
        med = np.median(cov)
        low_overall = float((cov <= med).mean())
        low_hard = float((cov[hard] <= med).mean())
        # also bottom-quartile share
        q1 = np.quantile(cov, 0.25)
        botq_overall = float((cov <= q1).mean())
        botq_hard = float((cov[hard] <= q1).mean())
        print(f"  [{tag}] among top-10% hardest positives (n={k}): "
              f"low-cov(<=median) frac={low_hard:.3f} vs overall {low_overall:.3f}  "
              f"|| bottom-quartile frac={botq_hard:.3f} vs overall {botq_overall:.3f} "
              f"(expect hard >> overall)")

    # =================================================================== #
    # DESTRUCTIVE CONTROLS D1-D5
    # =================================================================== #
    print("\n" + "#" * 78)
    print("DESTRUCTIVE CONTROLS  (each decides novelty)")
    print("#" * 78)

    # D1: permute cov -> correlation must vanish
    print("\n-- D1: permute cov (shuffle) -> rho must vanish --")
    for tag, cov in (('cov2d', c2), ('cov3d', c3)):
        rho_real, _ = spearman(cos, cov)
        sh = RNG.permutation(cov)
        rho_sh, _ = spearman(cos, sh)
        print(f"  [{tag}] rho(cos,cov)={rho_real:+.4f}  ->  rho(cos, shuffled cov)={rho_sh:+.4f} "
              f"(must -> ~0)")

    # D2: bbox-area-ratio / camera-pair as overlap proxy -> SMPL cov must explain more
    print("\n-- D2: bbox area-ratio & camera-pair as overlap proxies (SMPL cov must beat) --")
    # area-ratio in [0,1] (smaller = more scale mismatch). Use as 'overlap-like' signal.
    rho_area, _ = spearman(cos, arearatio)              # more scale-match -> smaller dist?
    print(f"  rho(cos, bbox-area-ratio)         = {rho_area:+.4f}")
    # camera-pair: how much variance in cos does camera-pair explain (eta^2)?
    eta2_cam = _eta_squared(cos, campair)
    print(f"  eta^2(cos ~ camera-pair)          = {eta2_cam:.4f}")
    eta2_cov3 = _eta_squared(cos, quartile_buckets(c3)[0])
    eta2_cov2 = _eta_squared(cos, quartile_buckets(c2)[0])
    print(f"  eta^2(cos ~ cov2d-quartile)       = {eta2_cov2:.4f}")
    print(f"  eta^2(cos ~ cov3d-quartile)       = {eta2_cov3:.4f}")
    # partial: cov vs cos controlling for area-ratio
    for tag, cov in (('cov2d', c2), ('cov3d', c3)):
        rho_pa, _ = partial_spearman(cos, cov, arearatio)
        print(f"  partial rho(cos, {tag} | area-ratio) = {rho_pa:+.4f} (must stay negative)")

    # D3: feature-distance as difficulty -> cov must add signal in partial correlation.
    # (cos IS the feature distance, so we test that cov predicts AP beyond feature-dist:
    #  partial rho(AP, max-cov | query mean-cos-to-truematch).)
    print("\n-- D3: SMPL cov beyond feature-distance (partial corr controlling cos) --")
    # per-query difficulty = mean cosine distance to its true matches
    qdiff = np.full(len(q['name']), np.nan)
    bymc = defaultdict(list)
    for p in P:
        bymc[p['qi']].append(p['cos'])
    for i in bymc:
        qdiff[i] = np.mean(bymc[i])
    for tag, mc in (('cov2d', q_maxcov2), ('cov3d', q_maxcov3)):
        sel = (aps >= 0) & np.isfinite(mc) & np.isfinite(qdiff)
        rho_raw, _ = spearman(mc[sel], aps[sel])
        rho_par, _ = partial_spearman(aps[sel], mc[sel], qdiff[sel])
        print(f"  [{tag}] rho(max-cov, AP)={rho_raw:+.4f}  "
              f"partial(max-cov, AP | mean-cos-to-truematch)={rho_par:+.4f} "
              f"(must stay positive)")
    # also pair-level: partial rho(cos, cov | nvis) already in Test 1; report a direct one:
    for tag, cov in (('cov2d', c2), ('cov3d', c3)):
        rho_par2, _ = partial_spearman(cos, cov, nvis)
        print(f"  [{tag}] pair-level partial rho(cos, cov | #vis-joints)={rho_par2:+.4f}")

    # D4: cov2d vs cov3d head-to-head — THE make-or-break line
    print("\n" + "=" * 78)
    print("D4 (CRITICAL) — cov3d must clearly BEAT cov2d")
    print("=" * 78)
    rho2, _ = spearman(cos, c2)
    rho3, _ = spearman(cos, c3)
    rho3p, _ = spearman(cos, c3p)
    # partial: does cov3d explain cos BEYOND cov2d? and vice versa?
    rho3_given2, _ = partial_spearman(cos, c3, c2)
    rho2_given3, _ = partial_spearman(cos, c2, c3)
    # non-circular variant: front-facing-ONLY cov3d_pure (does NOT AND with 2D in-frame)
    rho3p_given2, _ = partial_spearman(cos, c3p, c2)
    print(f"  rho(cos, cov2d)                 = {rho2:+.4f}")
    print(f"  rho(cos, cov3d)  [front&inframe]= {rho3:+.4f}")
    print(f"  rho(cos, cov3d_pure) [front-only,2D-independent] = {rho3p:+.4f}")
    print(f"  partial rho(cos, cov3d | cov2d) = {rho3_given2:+.4f}  "
          f"(cov3d's UNIQUE signal beyond 2D)")
    print(f"  partial rho(cos, cov2d | cov3d) = {rho2_given3:+.4f}  "
          f"(cov2d's UNIQUE signal beyond 3D)")
    print(f"  partial rho(cos, cov3d_pure | cov2d) = {rho3p_given2:+.4f}  "
          f"(non-circular: front-facing signal beyond 2D in-frame)")
    # bucket-gap comparison
    def bucket_gap(cov):
        idx, _, _ = quartile_buckets(cov)
        m = [cos[idx == b].mean() if (idx == b).sum() else np.nan for b in range(4)]
        return m[0] - m[3]
    print(f"  Q0-Q3 cosine-dist gap: cov2d={bucket_gap(c2):+.4f}  cov3d={bucket_gap(c3):+.4f}")
    verdict_d4 = (abs(rho3) > abs(rho2) + 0.01) and (abs(rho3_given2) > abs(rho2_given3))
    print(f"  >> D4 verdict: cov3d {'BEATS' if verdict_d4 else 'does NOT clearly beat'} cov2d "
          f"(|rho3|>|rho2| AND unique3>unique2)")

    # D5: random joint subset as 'visible' -> correlation must drop
    print("\n-- D5: random joint subset as 'visible' (must drop vs real cov2d) --")
    # for each image pick a random subset of joints of the SAME SIZE as its real vis2d,
    # recompute cov, correlate.
    def random_cov(qv, gv, valid_q, valid_g):
        out = np.zeros(n)
        for t, p in enumerate(P):
            si = q_si[p['qi']]; sj = g_si[p['gj']]
            ka = int(qv[si].sum()); kb = int(gv[sj].sum())
            ra = np.zeros(71, bool); rb = np.zeros(71, bool)
            if ka > 0:
                ra[RNG.choice(71, ka, replace=False)] = True
            if kb > 0:
                rb[RNG.choice(71, kb, replace=False)] = True
            out[t] = cov_iou(ra, rb)
        return out
    rc = random_cov(qv2, gv2, qvalid, gvalid)
    rho_rc, _ = spearman(cos, rc)
    print(f"  rho(cos, random-subset cov) = {rho_rc:+.4f}  (vs real cov2d {rho2:+.4f}; must -> ~0)")

    # =================================================================== #
    # FINAL VERDICT
    # =================================================================== #
    print("\n" + "#" * 78)
    print("SUMMARY / VERDICT (per GOPL_KILLSWITCH_DESIGN pass criteria)")
    print("#" * 78)
    print(f"  sanity frozen mAP                = {res['mAP']:.2f}")
    print(f"  T1 rho(cos,cov2d)/cov3d          = {rho2:+.4f} / {rho3:+.4f}")
    print(f"  T1 partial(cos,cov|#vis) 2d/3d   = "
          f"{partial_spearman(cos,c2,nvis)[0]:+.4f} / {partial_spearman(cos,c3,nvis)[0]:+.4f}")
    print(f"  D1 shuffle 3d                    -> {spearman(cos, RNG.permutation(c3))[0]:+.4f} (want ~0)")
    print(f"  D2 partial(cos,cov3d|area)       = {partial_spearman(cos,c3,arearatio)[0]:+.4f}")
    print(f"  D4 |rho3|>|rho2| & unique3>unique2 = {verdict_d4}")
    print(f"  D5 random-subset rho             = {rho_rc:+.4f} (want ~0)")
    # CORE requires: cov3d predicts cos-dist, survives shuffle, AND survives the
    # random-subset (size) control AND retains signal after removing #vis-count.
    cov3d_after_nvis = partial_spearman(cos, c3, nvis)[0]
    core_ok = (rho3 < -0.03) and (abs(rho_rc) < abs(rho3) * 0.5) and (cov3d_after_nvis < -0.03)
    print(f"\n  >>> CORE (cov3d predicts cos-dist & survives shuffle/random-subset/#vis-count): "
          f"{'PASS' if core_ok else 'FAIL'}")
    print(f"      [random-subset rho {rho_rc:+.4f} vs real {rho3:+.4f}; "
          f"cov3d-after-#vis {cov3d_after_nvis:+.4f}]")
    print(f"  >>> D4 (cov3d > cov2d, GOPL vs VPM/QPM): {'PASS' if verdict_d4 else 'FAIL — GOPL downgrades'}")
    print("\n[done] GOPL kill-switch complete.")


# ---- small utilities (areas, ratios, eta^2) ---- #
_AREA_CACHE = {}
def _areas(paths, fallback_dir=None):
    from PIL import Image
    out = np.full(len(paths), -1.0)
    for i, p in enumerate(paths):
        if p in _AREA_CACHE:
            out[i] = _AREA_CACHE[p]; continue
        a = -1.0
        try:
            w, h = Image.open(p).size; a = float(w * h)
        except Exception:
            if fallback_dir is not None:
                try:
                    w, h = Image.open(os.path.join(fallback_dir, os.path.basename(p))).size
                    a = float(w * h)
                except Exception:
                    a = -1.0
        _AREA_CACHE[p] = a; out[i] = a
    return out


def _ratio(a, b):
    if a <= 0 or b <= 0:
        return np.nan
    r = a / b
    return r if r <= 1 else 1.0 / r   # symmetric, in (0,1], 1=same scale


def _eta_squared(y, groups):
    """Fraction of variance in y explained by categorical 'groups' (one-way ANOVA eta^2)."""
    y = np.asarray(y, float)
    g = np.asarray(groups)
    grand = y.mean()
    ss_tot = ((y - grand) ** 2).sum()
    ss_between = 0.0
    for gv in np.unique(g):
        m = g == gv
        if m.sum() == 0:
            continue
        ss_between += m.sum() * (y[m].mean() - grand) ** 2
    return float(ss_between / ss_tot) if ss_tot > 0 else 0.0


if __name__ == '__main__':
    main()
