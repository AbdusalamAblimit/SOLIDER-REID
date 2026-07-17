#!/usr/bin/env python3
"""FGEU REALIZABILITY kill-switch — DECISIVE same-camera-tracklet vs cross-camera-oracle.

THE LIFE-OR-DEATH QUESTION
--------------------------
FGEU (Fragility-Guided Evidence Union): for an evidence-insufficient (occluded) query,
union ADDITIONAL query-side evidence to recover retrieval. Zero-training already proved
ORACLE same-ID evidence union has large headroom (OD 17->52, beats k-reciprocal 3.3x).
BUT that union used CROSS-CAMERA same-ID images (oracle, NOT deployable = the exp109 wall).

The命门 (crux): does a *REALIZABLE* SAME-CAMERA tracklet (deployable: continuous frames of
the same person from one camera, available at query time WITHOUT identity labels) recover
retrieval too? If yes -> FGEU is a real, deployable method (7/10). If only the cross-camera
ORACLE has headroom and the realizable same-camera tracklet does not -> exp109 query-side
wall, FGEU dies (3/10).

DATA: occluded_posetrack_reid. Filenames `pid_cVID_TIMESTAMP.jpg` (e.g. 0000_c68_1008827004807).
  * same pid + same cVID = same VIDEO/tracklet (continuous frames, ONE camera) = REALIZABLE evidence.
  * same pid + DIFFERENT cVID (or a gallery img from another video) = cross-video = ORACLE headroom.
NOTE: the repo's posetrack loader assigns a per-IMAGE unique camid (junk filter is a no-op,
KPR mot_inter_intra_video protocol). So the cached q_cam/g_cam are NOT the tracklet id — we
recover the tracklet from the FILENAME's c{VID} field. (We still pass per-image unique cams to
per_query_ap so the same-pid&cam junk removal stays a no-op, matching the training eval.)

ARMS (all on FROZEN features, numpy only, NO backward):
  baseline : single-frame AP per query (the first frame of each tracklet).
  A_realizable : union the EXTRA same-(pid,video) frame(s) — deployable. mean & max pool.
  B_oracle     : union a CROSS-VIDEO same-ID *gallery* image — NOT deployable (exp109 upper bound).
  C_controls   : mean/max fusion (== A, reported as the trivial multi-frame baseline),
                 k-reciprocal re-rank on the SINGLE frame (free re-rank, no new evidence),
                 RANDOM cross-ID frame union (must DESTROY -> proves gain is identity evidence).
  Recovery ratios: dAP_realizable / dAP_oracle.
  Fragility gate : fuse only when lowtail-positive-support is weak (fragile) vs fuse-all.

VERDICT: same-camera tracklet union (REALIZABLE) attains >=40% of the oracle recovery AND
clearly beats k-reciprocal / mean-fusion -> FGEU deployable (7/10). Only cross-camera oracle
has headroom, realizable cannot -> exp109 query-side wall (3/10, DEAD). Be honest.

Run on lab-3090-d:
  cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 python3 \
    experiments/cargo_cvpb/cvpb_realizability_killswitch.py \
    --config configs/occluded_posetrack/prcv_best_base.yml \
    --ckpt log/occluded_posetrack/exp266b_best_b_op_s41_3090/transformer_120.pth \
    2>&1 | tee /tmp/cvpb_realizability.log
  # smoke first: add  --smoke 300   ;  reuse feats: --reuse_feat
"""
import os, sys, time, argparse, re
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.abspath(os.path.join(_here, '..', '..'))
sys.path.insert(0, _repo)

ap = argparse.ArgumentParser()
ap.add_argument('--config', default='configs/occluded_posetrack/prcv_best_base.yml')
ap.add_argument('--ckpt', default='log/occluded_posetrack/exp266b_best_b_op_s41_3090/transformer_120.pth')
ap.add_argument('--cache_feat', default='/tmp/realiz_posetrack_feats.npz')
ap.add_argument('--reuse_feat', action='store_true')
ap.add_argument('--smoke', type=int, default=0, help='cap #query for a fast smoke run')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--a_temp', type=float, default=20.0, help='soft-min temp for lowtail positive support')
ap.add_argument('--fail_quant', type=float, default=0.50,
                help='per-tracklet single-frame AP bottom-q = failure subset (the rescue target)')
ap.add_argument('--frag_quant', type=float, default=0.50,
                help='fragility gate: among failures, bottom-q lowtail-support = "fragile" (fuse only these)')
ap.add_argument('--n_rand', type=int, default=5, help='#random cross-ID partners averaged for the control')
ap.add_argument('--krecip_k1', type=int, default=20)
ap.add_argument('--krecip_k2', type=int, default=6)
ap.add_argument('--krecip_lambda', type=float, default=0.3)
cli = ap.parse_args()
np.random.seed(cli.seed)
RNG = np.random.RandomState(cli.seed)


# =========================================================================== #
# 1. FEATURE EXTRACTION (frozen posetrack ckpt, equal_concat = the real eval feat)
#    Reuses the validated extraction path from cvpb_gopl_killswitch.py.
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
        'TEST.NECK_FEAT', cfg.TEST.NECK_FEAT,      # keep config's trained eval convention
        'TEST.FEAT_NORM', 'yes',
        'TEST.IMS_PER_BATCH', 64,
    ])
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    (train_loader, train_loader_normal, val_loader, num_query,
     num_classes, camera_num, view_num) = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num,
                       view_num=view_num, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
    model.load_param(cfg.TEST.WEIGHT)
    model = model.cuda().eval()
    print(f"[extract] loaded {cli.ckpt}; POSE_TEST_FEAT={cfg.MODEL.POSE_TEST_FEAT}; "
          f"num_query={num_query}", flush=True)

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
                f"expected 2D feature, got {type(feat)} {getattr(feat,'shape',None)}"
            feat = F.normalize(feat, p=2, dim=1)
            feats.append(feat.cpu().numpy().astype(np.float32))
            pids.extend([int(x) for x in b_pids])
            camids.extend([int(x) for x in b_camids_t.tolist()])
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


def load_or_extract():
    if cli.reuse_feat and os.path.exists(cli.cache_feat):
        z = np.load(cli.cache_feat, allow_pickle=True)
        q = dict(feat=z['q_feat'], pid=z['q_pid'], cam=z['q_cam'], name=z['q_name'])
        g = dict(feat=z['g_feat'], pid=z['g_pid'], cam=z['g_cam'], name=z['g_name'])
        print(f"[data] reused {cli.cache_feat}", flush=True)
    else:
        q, g = extract_features()
    for d in (q, g):
        d['feat'] = d['feat'].astype(np.float32)
        d['feat'] /= (np.linalg.norm(d['feat'], axis=1, keepdims=True) + 1e-12)
        # video_id parsed from filename c{VID} (the TRACKLET id; cached cam is per-image unique)
        vids = []
        for nm in d['name']:
            m = re.search(r'_c(\d+)_', str(nm))
            vids.append(int(m.group(1)) if m else -1)
        d['vid'] = np.asarray(vids)
    print(f"[data] Nq={len(q['name'])} Ng={len(g['name'])} dim={q['feat'].shape[1]} "
          f"#q-pid={len(np.unique(q['pid']))} #g-pid={len(np.unique(g['pid']))} "
          f"#q-vid={len(np.unique(q['vid']))}", flush=True)
    return q, g


# =========================================================================== #
# per-query AP. dist = 1 - cosine. junk removal (same pid & cam) is a no-op here
# because posetrack cams are per-image unique (matches the training eval protocol).
# =========================================================================== #
def per_query_ap(qf, gf, q_pid, q_cam, g_pid, g_cam):
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


def per_query_ap_from_dist(distmat, sel_rows, q_pid, q_cam, g_pid, g_cam):
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


def softmin(s, a):
    s = np.asarray(s, float)
    if len(s) == 0:
        return np.nan
    m = (-a * s).max()
    return -(m + np.log(np.exp(-a * s - m).sum())) / a


def kreciprocal_rerank(qf, gf, k1=20, k2=6, lam=0.3):
    """Standard k-reciprocal re-ranking (Zhong CVPR2017), verbatim from the validated impl."""
    qf = qf.astype(np.float32); gf = gf.astype(np.float32)
    Nq, Ng = qf.shape[0], gf.shape[0]
    allf = np.concatenate([qf, gf], 0)
    orig = np.maximum(2.0 - 2.0 * (allf @ allf.T), 0.0)
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


# =========================================================================== #
# Tracklet bookkeeping: group query frames by (pid, video_id). The FIRST frame
# (earliest timestamp by name-sort order = dataloader order) is the deployed single
# query; the rest are REALIZABLE extra evidence.
# =========================================================================== #
def build_tracklets(q):
    pid, vid, name = q['pid'], q['vid'], q['name']
    order = np.argsort(name)            # stable, == loader sort order (filenames)
    trk = {}
    for idx in order:
        key = (int(pid[idx]), int(vid[idx]))
        trk.setdefault(key, []).append(int(idx))
    sizes = np.array([len(v) for v in trk.values()])
    from collections import Counter
    print(f"[trk] #query tracklets (pid,video)={len(trk)}  size-hist={dict(sorted(Counter(sizes).items()))}  "
          f">=2 frames (realizable union avail)={int((sizes>=2).sum())}", flush=True)
    return trk


# =========================================================================== #
# MAIN
# =========================================================================== #
def main():
    print("#" * 84)
    print(f"# FGEU REALIZABILITY KILL-SWITCH  (occluded_posetrack)  ckpt={os.path.basename(cli.ckpt)}")
    print("#" * 84)
    t0 = time.time()
    q, g = load_or_extract()
    qf, q_pid, q_cam, q_vid, q_name = q['feat'], q['pid'], q['cam'], q['vid'], q['name']
    gf, g_pid, g_cam, g_vid = g['feat'], g['pid'], g['cam'], g['vid']

    # ---- SANITY: full-gallery mAP with the standard (all query frames) protocol ----
    base_all = per_query_ap(qf, gf, q_pid, q_cam, g_pid, g_cam)
    val = base_all >= 0
    print(f"[SANITY] frozen full-query mAP={base_all[val].mean()*100:.2f}  nq={int(val.sum())}  "
          f"(in-domain train log reported 78.5)", flush=True)

    trk = build_tracklets(q)

    # ---- single-frame deploy: first frame of each tracklet = the deployed query ----
    # AP of that single frame on the full gallery.
    single_rows, extra_rows = [], {}     # extra_rows[first_idx] = [other frame idxs]
    for key, idxs in trk.items():
        single_rows.append(idxs[0])
        extra_rows[idxs[0]] = idxs[1:]
    single_rows = np.array(sorted(single_rows))
    ap_single = per_query_ap(qf[single_rows], gf, q_pid[single_rows], q_cam[single_rows], g_pid, g_cam)
    oks = ap_single >= 0
    print(f"\n[single] single-frame (1 per tracklet) mAP={ap_single[oks].mean()*100:.3f}  "
          f"n_tracklets={int(oks.sum())}", flush=True)

    # lowtail positive support of each single frame (cross-VIDEO positives in gallery only;
    # weak = fragile). Used for the fragility gate. We use cross-video gallery positives so the
    # support meter reflects deployable cross-camera identity evidence, not same-video repeats.
    sim_qg = qf[single_rows] @ gf.T
    lowtail = np.full(len(single_rows), np.nan)
    for r, i in enumerate(single_rows):
        keep_pos = (g_pid == q_pid[i]) & (g_vid != q_vid[i])      # cross-video same-ID gallery
        s = sim_qg[r][keep_pos]
        if len(s):
            lowtail[r] = softmin(s, cli.a_temp)

    # ---- FAILURE subset: bottom-q single-frame AP (the rescue target) ----
    vidx = np.where(oks)[0]
    nfail = int(round(cli.fail_quant * len(vidx)))
    jit = np.random.RandomState(cli.seed + 3).rand(len(vidx)) * 1e-9
    order = np.argsort(ap_single[single_rows][vidx] + jit if False else ap_single[vidx] + jit)
    fail_local = vidx[order[:nfail]]                  # indices into single_rows
    print(f"[fail] failure subset = bottom-{cli.fail_quant:.0%} single-frame AP: "
          f"n={len(fail_local)} (mean AP={ap_single[fail_local].mean()*100:.3f})", flush=True)

    # restrict to failures that HAVE a realizable extra same-video frame
    fail_realiz = [r for r in fail_local if len(extra_rows[single_rows[r]]) >= 1]
    print(f"[fail] of which have >=1 realizable same-video extra frame: {len(fail_realiz)}", flush=True)

    # =====================================================================
    # ARM A (REALIZABLE): union the extra same-(pid,video) frame(s) — deployable
    # ARM B (ORACLE):     union a cross-video same-ID GALLERY image — exp109 upper bound
    # ARM C (CONTROLS):   k-reciprocal (single frame) + random cross-ID frame union
    # =====================================================================
    rs = np.random.RandomState(cli.seed + 17)
    base_l, A_mean_l, A_max_l, B_orac_l, C_rand_l = [], [], [], [], []
    used_rows = []
    n_no_oracle = 0
    # build pid -> cross-video gallery indices for the oracle
    for r in fail_realiz:
        i = single_rows[r]
        extra = extra_rows[i]                          # realizable same-video extra frames
        # ---- baseline AP (single frame) ----
        ap_b = per_query_ap(qf[i:i+1], gf, q_pid[i:i+1], q_cam[i:i+1], g_pid, g_cam)[0]
        # ---- A_realizable: union single + same-video extra frame(s) ----
        pack = np.concatenate([qf[i][None], qf[extra]], 0)      # (1+k, D)
        f_mean = pack.sum(0); f_mean /= (np.linalg.norm(f_mean) + 1e-12)
        f_max = pack.max(0);  f_max /= (np.linalg.norm(f_max) + 1e-12)
        ap_Am = per_query_ap(f_mean[None], gf, q_pid[i:i+1], q_cam[i:i+1], g_pid, g_cam)[0]
        ap_Ax = per_query_ap(f_max[None],  gf, q_pid[i:i+1], q_cam[i:i+1], g_pid, g_cam)[0]
        # ---- B_oracle: union single + a cross-video same-ID GALLERY image (NOT deployable) ----
        gcross = np.where((g_pid == q_pid[i]) & (g_vid != q_vid[i]))[0]
        if len(gcross) == 0:
            ap_B = np.nan
            n_no_oracle += 1
        else:
            gj = gcross[rs.randint(len(gcross))]
            f_or = qf[i] + gf[gj]; f_or /= (np.linalg.norm(f_or) + 1e-12)
            ap_B = per_query_ap(f_or[None], gf, q_pid[i:i+1], q_cam[i:i+1], g_pid, g_cam)[0]
        # ---- C_rand: union single + RANDOM cross-ID query frame (must DESTROY), avg draws ----
        rand_draws = []
        for _ in range(cli.n_rand):
            while True:
                jr = rs.randint(len(q_pid))
                if int(q_pid[jr]) != int(q_pid[i]):
                    break
            f_rd = qf[i] + qf[jr]; f_rd /= (np.linalg.norm(f_rd) + 1e-12)
            rand_draws.append(per_query_ap(f_rd[None], gf, q_pid[i:i+1], q_cam[i:i+1], g_pid, g_cam)[0])
        base_l.append(ap_b); A_mean_l.append(ap_Am); A_max_l.append(ap_Ax)
        B_orac_l.append(ap_B); C_rand_l.append(float(np.mean(rand_draws)))
        used_rows.append(r)

    base_a = np.array(base_l); Am = np.array(A_mean_l); Ax = np.array(A_max_l)
    Bo = np.array(B_orac_l); Cr = np.array(C_rand_l)
    n = len(base_a)
    okB = np.isfinite(Bo)
    print("\n" + "=" * 84)
    print(f"# CORE LIFE-OR-DEATH TABLE  (n={n} realizable-failure tracklets; "
          f"oracle avail on {int(okB.sum())})")
    print("=" * 84)
    print(f"  baseline single-frame AP            = {base_a.mean()*100:.3f}")
    print(f"  A_realizable union same-video MEAN  = {Am.mean()*100:.3f}  "
          f"(dAP={ (Am.mean()-base_a.mean())*100:+.3f})")
    print(f"  A_realizable union same-video MAX   = {Ax.mean()*100:.3f}  "
          f"(dAP={ (Ax.mean()-base_a.mean())*100:+.3f})")
    print(f"  B_oracle union cross-video GALLERY  = {Bo[okB].mean()*100:.3f}  "
          f"(dAP={ (Bo[okB].mean()-base_a[okB].mean())*100:+.3f})  [NOT deployable; exp109 upper bound]")
    print(f"  C_rand cross-ID union (CONTROL)     = {Cr.mean()*100:.3f}  "
          f"(dAP={ (Cr.mean()-base_a.mean())*100:+.3f})  [MUST destroy]")

    # recovery ratios on the COMMON oracle-available subset (apples-to-apples)
    if okB.sum() > 0:
        dA = (Am[okB].mean() - base_a[okB].mean()) * 100
        dAx = (Ax[okB].mean() - base_a[okB].mean()) * 100
        dB = (Bo[okB].mean() - base_a[okB].mean()) * 100
        ratio_mean = dA / dB if abs(dB) > 1e-6 else float('nan')
        ratio_max = dAx / dB if abs(dB) > 1e-6 else float('nan')
        print(f"\n  [on the {int(okB.sum())} oracle-available tracklets]")
        print(f"  dAP realizable-MEAN = {dA:+.3f} | dAP realizable-MAX = {dAx:+.3f} | "
              f"dAP oracle = {dB:+.3f}")
        print(f"  >> Recovery_realizable/Recovery_oracle  MEAN={ratio_mean*100:.1f}%  "
              f"MAX={ratio_max*100:.1f}%")
    # recovery RATE (fraction of queries improved by a meaningful margin)
    print(f"\n  recovery-rate (dAP>+0.05): A_realizable-mean={float((Am-base_a>0.05).mean()):.3f}  "
          f"oracle={float((Bo[okB]-base_a[okB]>0.05).mean()) if okB.sum() else float('nan'):.3f}  "
          f"random={float((Cr-base_a>0.05).mean()):.3f}")

    # =====================================================================
    # C: k-reciprocal on the SINGLE frame (free re-rank, no new evidence)
    # run on the full single-frame query set, index the used rows.
    # =====================================================================
    print("\n" + "-" * 84)
    print(f"[C k-recip] k-reciprocal on SINGLE frames (k1={cli.krecip_k1} k2={cli.krecip_k2} "
          f"lam={cli.krecip_lambda}); index the {n} used failure tracklets:")
    try:
        rr = kreciprocal_rerank(qf[single_rows], gf, cli.krecip_k1, cli.krecip_k2, cli.krecip_lambda)
        base_dist = 1.0 - qf[single_rows] @ gf.T
        used_arr = np.array(used_rows)
        ap_kr = per_query_ap_from_dist(rr, used_arr, q_pid[single_rows], q_cam[single_rows], g_pid, g_cam)
        ap_brows = per_query_ap_from_dist(base_dist, used_arr, q_pid[single_rows], q_cam[single_rows],
                                          g_pid, g_cam)
        ok = (ap_kr >= 0) & (ap_brows >= 0)
        dkr = (ap_kr[ok].mean() - ap_brows[ok].mean()) * 100
        print(f"     base AP (these rows) = {ap_brows[ok].mean()*100:.3f}  (vs single-probe loop "
              f"base={base_a.mean()*100:.3f})")
        print(f"     k-reciprocal AP      = {ap_kr[ok].mean()*100:.3f}  (dAP={dkr:+.3f})")
        print(f"     >> A_realizable-mean dAP={ (Am.mean()-base_a.mean())*100:+.3f}  vs  "
              f"k-recip dAP={dkr:+.3f}  [union beats free re-rank only if >>]")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"     [k-reciprocal failed: {e}]")

    # =====================================================================
    # Fragility gate: fuse only FRAGILE (weak lowtail support) failures vs fuse-all.
    # Among the used failures, split by lowtail support (computed earlier per single_rows row).
    # =====================================================================
    print("\n" + "-" * 84)
    print(f"[gate] fragility-weighted (fuse only weak-support failures) vs fuse-all")
    lt_used = lowtail[np.array(used_rows)]
    okg = np.isfinite(lt_used)
    if okg.sum() >= 6:
        thr = np.quantile(lt_used[okg], cli.frag_quant)
        fragile = okg & (lt_used <= thr)               # weakest support = most fragile
        nonfrag = okg & (lt_used > thr)
        dA_all = (Am.mean() - base_a.mean()) * 100
        dA_frag = (Am[fragile].mean() - base_a[fragile].mean()) * 100 if fragile.sum() else float('nan')
        dA_non = (Am[nonfrag].mean() - base_a[nonfrag].mean()) * 100 if nonfrag.sum() else float('nan')
        print(f"     fuse-ALL  dAP={dA_all:+.3f} (n={n})")
        print(f"     fuse-FRAGILE-only (bottom-{cli.frag_quant:.0%} support) dAP={dA_frag:+.3f} "
              f"(n={int(fragile.sum())})")
        print(f"     non-fragile dAP={dA_non:+.3f} (n={int(nonfrag.sum())})")
        print(f"     >> gate helps if fragile dAP > non-fragile dAP (evidence union targets fragile)")
    else:
        print(f"     [too few finite-support rows to gate: {int(okg.sum())}]")

    # =====================================================================
    # VERDICT
    # =====================================================================
    print("\n" + "#" * 84)
    print(f"# VERDICT  ({time.time()-t0:.0f}s)")
    print("#" * 84)
    if okB.sum() > 0:
        dA = (Am[okB].mean() - base_a[okB].mean()) * 100
        dB = (Bo[okB].mean() - base_a[okB].mean()) * 100
        ratio = dA / dB if abs(dB) > 1e-6 else float('nan')
        rand_destroys = (Cr.mean() - base_a.mean()) < 0  # random must not help
        print(f"  realizable dAP={dA:+.3f}  oracle dAP={dB:+.3f}  ratio={ratio*100:.1f}%  "
              f"(threshold 40%)")
        print(f"  random-control dAP={ (Cr.mean()-base_a.mean())*100:+.3f}  "
              f"({'DESTROYS (good)' if rand_destroys else 'DID NOT destroy (suspicious)'})")
        live = (ratio >= 0.40) and rand_destroys and (dA > 0)
        print(f"\n  [{'LIVE 7/10' if live else 'DEAD 3/10 — exp109 query-side wall'}] "
              f"realizable same-camera tracklet union "
              f"{'attains >=40% of oracle recovery' if live else 'does NOT attain 40% of oracle / no headroom'}")
    else:
        print("  [no oracle-available tracklets] cannot compute realizable/oracle ratio; "
              "report realizable dAP only.")
    print("# (final call is human; read the raw numbers above)")
    print("[done]")


if __name__ == '__main__':
    main()
