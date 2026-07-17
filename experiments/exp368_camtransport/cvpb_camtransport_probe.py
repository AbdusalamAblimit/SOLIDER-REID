#!/usr/bin/env python3
"""Camera-Pose Transport cheap kill-switch (frozen, 零训练) — codex 训练侧 Top3.

不学 camera-invariant descriptor, 而学 low-rank transport 把 descriptor 从一个 camera
cell 映到另一个 cell 后再比较 (comparability operator, not invariance)。

cheap kill-switch:
  frozen global feat (ae_feats.npz, query/gallery + cam) +
  train ID 拟合 cam pair ID-mean ridge transport W_{a→b} (cam a feat → cam b 空间) +
  test query(cam a) transport 到 gallery cam b 后 cosine。
成功线: 同 #false@10 分桶 mAP +0.5 (vs 直接 cosine), camera-centering 弱对照不应同等增益。
若 transport 抬 → camera invariance 不够, transport 有 headroom; 不抬 → DEAD 转 Top2。

frozen 零训练 (豁免训练审查)。Run on lab-3090-d: cd repo && python experiments/exp368_camtransport/cvpb_camtransport_probe.py
"""
import os, sys, argparse
import numpy as np
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('--eval_cache', default='/tmp/ae_feats.npz')
ap.add_argument('--train_cache', default='/tmp/camtransport_train.npz')
ap.add_argument('--lam', type=float, default=1.0)      # ridge 正则
ap.add_argument('--ckpt', default='log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth')
ap.add_argument('--config', default='configs/market/pose_psg_lgpa_gcn_base.yml')
cli = ap.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'experiments', 'cargo_cvpb'))
import cvpb_lattice_killswitch as ks
from datasets.bases import read_image
REPO = ks._repo


def l2n(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


# ---- 1. train feat + cam (frozen) ----
if os.path.exists(cli.train_cache):
    z = np.load(cli.train_cache); tf, tp, tc = z['tf'], z['tp'], z['tc']
    print(f'[train feat] cached {tf.shape}', flush=True)
else:
    sys.argv = ['ks', '--ckpt', cli.ckpt, '--config', cli.config, '--data_root', 'data', '--K', '1',
                '--reuse_gallery', '--cache_gallery', '/tmp/ct_g.npz']
    ext = ks.FrozenExtractor()
    its = ks.list_split(os.path.join(REPO, 'data', 'market1501', 'bounding_box_train'))
    pils = [ks._to_target_aspect(read_image(it[0])) for it in its]
    tf = l2n(ext.feats_from_pil(pils).astype(np.float32))
    tp = np.array([it[1] for it in its]); tc = np.array([it[2] for it in its])
    np.savez(cli.train_cache, tf=tf, tp=tp, tc=tc); print(f'[train feat] extracted {tf.shape}', flush=True)

D = tf.shape[1]
cams = sorted(set(tc.tolist()))
id2idx = defaultdict(list)
for i, p in enumerate(tp): id2idx[int(p)].append(i)
ids = list(id2idx.keys())
print(f'[camtransport] D={D} cams={cams} train-ids={len(ids)}', flush=True)

# ---- 2. cam pair transport map (ID-mean ridge): cam a feat → cam b ----
# 对每对 (a,b): 取在 a,b 都有图的 ID, 用 ID-mean feat 对 (Xa, Xb) ridge 学 W: Xa @ W ≈ Xb
W = {a: {} for a in cams}
for a in cams:
    for b in cams:
        if a == b:
            W[a][b] = np.eye(D, dtype=np.float32); continue
        Xa, Xb = [], []
        for i in ids:
            idx = np.array(id2idx[i])
            ca = idx[tc[idx] == a]; cb = idx[tc[idx] == b]
            if len(ca) and len(cb):
                Xa.append(tf[ca].mean(0)); Xb.append(tf[cb].mean(0))
        if len(Xa) < D // 8:                          # 样本太少, 退化为 identity
            W[a][b] = np.eye(D, dtype=np.float32); continue
        Xa = np.stack(Xa); Xb = np.stack(Xb)
        W[a][b] = np.linalg.solve(Xa.T @ Xa + cli.lam * np.eye(D), Xa.T @ Xb).astype(np.float32)

# ---- 3. test: ae_feats query/gallery ----
z = np.load(cli.eval_cache)
qf, qp, qc, gf, gp, gc = z['qf'], z['qp'], z['qc'], z['gf'], z['gp'], z['gc']
qf, gf = l2n(qf.astype(np.float32)), l2n(gf.astype(np.float32))

# camera-centering 弱对照: per-cam mean 减 (train cam mean), 看 transport 是否只是去 camera bias
cam_mean = {a: tf[tc == a].mean(0) for a in cams}
gmean = np.stack([cam_mean.get(int(c), np.zeros(D)) for c in gc])
qmean = np.stack([cam_mean.get(int(c), np.zeros(D)) for c in qc])

valid_q = np.array([i for i in range(len(qf)) if (gp[~((gp == qp[i]) & (gc == qc[i]))] == qp[i]).any()], dtype=int)


def sim_baseline():
    return qf @ gf.T


def sim_centering():
    qc_ = l2n(qf - qmean); gc_ = l2n(gf - gmean)
    return qc_ @ gc_.T


def sim_transport():
    # query i (cam a=qc[i]) transport 到每个 gallery cam b: qf[i] @ W[a][b], 与 gallery-in-cam-b 比
    S = np.zeros((len(qf), len(gf)), dtype=np.float32)
    for a in cams:
        qa = np.where(qc == a)[0]
        if not len(qa): continue
        for b in cams:
            gb = np.where(gc == b)[0]
            if not len(gb): continue
            qt = l2n(qf[qa] @ W[a][b])
            S[np.ix_(qa, gb)] = qt @ gf[gb].T
    return S


def eval_sim(S, ref_false=None):
    aps, r1s, false10 = [], [], []
    for i in valid_q:
        keep = ~((gp == qp[i]) & (gc == qc[i]))
        s, gpk = S[i][keep], gp[keep]
        order = np.argsort(-s); match = (gpk[order] == qp[i])
        if not match.any():
            aps.append(0.0); r1s.append(0.0); false10.append(1.0); continue
        cum = np.cumsum(match); ranks = np.arange(1, len(match) + 1)
        aps.append((cum[match] / ranks[match]).mean()); r1s.append(float(match[0]))
        false10.append(float((gpk[order[:10]] != qp[i]).mean()))
    return np.array(aps), np.array(r1s), np.array(false10)


# baseline #false@10 分桶, 控 trivial 解释变量
ap_b, r1_b, f10_b = eval_sim(sim_baseline())
ap_c, r1_c, _ = eval_sim(sim_centering())
ap_t, r1_t, _ = eval_sim(sim_transport())

print(f'\n[CAMTRANSPORT RESULT]')
print(f'  baseline cosine  : mAP={100*ap_b.mean():.2f}  R1={100*r1_b.mean():.2f}')
print(f'  camera-centering : mAP={100*ap_c.mean():.2f}  R1={100*r1_c.mean():.2f}  Δ={100*(ap_c.mean()-ap_b.mean()):+.2f}')
print(f'  transport        : mAP={100*ap_t.mean():.2f}  R1={100*r1_t.mean():.2f}  Δ={100*(ap_t.mean()-ap_b.mean()):+.2f}')
# 按 baseline #false@10 分桶, 看 transport ΔAP 是否在难桶(false 多)上更显著, 而非 trivial
print(f'\n  [按 baseline #false@10 分桶 ΔAP(transport-baseline)]')
for lo, hi in [(0.0, 0.5), (0.5, 0.9), (0.9, 1.01)]:
    m = (f10_b >= lo) & (f10_b < hi)
    if m.sum() == 0: continue
    print(f'    false@10∈[{lo:.1f},{hi:.1f}) n={m.sum():4d}: baseAP={100*ap_b[m].mean():.2f} ΔAP={100*(ap_t[m]-ap_b[m]).mean():+.2f}')
print(f'\n  成功线: transport Δ>+0.5 且明显>camera-centering Δ → invariance 不够 transport 有 headroom', flush=True)
print('[done]', flush=True)
