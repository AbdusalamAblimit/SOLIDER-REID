#!/usr/bin/env python3
"""LPA head probe -- train-time mechanism A kill-switch.

Frozen no-LM-loss backbone. Train a tiny per-variant RELIABILITY head on the TRAIN ids
(predict each lattice variant's margin to its true id), then at TEST weight the K query
variants by softmax(head) and marginalize.  Train/test ids are DISJOINT (no leakage).

PASS  if  weighted-mAP - uniform-mAP >= +0.4  AND  head-argmax-variant matches the
per-query best-margin (oracle) variant >= 35% (chance = 1/K = 11%).  Else LPA learned head
is DEAD even though the oracle headroom is large -> the 'best variant' is not predictable.
Reuses cvpb_lattice_killswitch helpers via a sys.argv shim (its argparse is module-level).
"""
import sys, os, numpy as np, argparse, time

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='log/market1501/exp359_abl_noLMloss/transformer_40.pth')
ap.add_argument('--config', default='configs/market/pose_psg_lgpa_gcn_base.yml')
ap.add_argument('--h', type=int, default=16)
ap.add_argument('--K', type=int, default=9)
ap.add_argument('--epochs', type=int, default=40)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--tau', type=float, default=0.5)
ap.add_argument('--train_cap', type=int, default=0)        # 0 = all train imgs
ap.add_argument('--cache_gallery', default='/tmp/g_lpa_head.npz')
lpa = ap.parse_args()

# the kill-switch parses argv at import time -> feed it the args its FrozenExtractor needs.
sys.argv = ['ks', '--ckpt', lpa.ckpt, '--config', lpa.config, '--data_root', 'data',
            '--K', str(lpa.K), '--reuse_gallery', '--cache_gallery', lpa.cache_gallery]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvpb_lattice_killswitch as ks
import torch
from datasets.bases import read_image

RNG = np.random.RandomState(42)
H, K = lpa.h, lpa.K
ext = ks.FrozenExtractor()
REPO = ks._repo


def items(split):
    return ks.list_split(os.path.join(REPO, 'data', 'market1501', split))


def variant_feats(its, cap=0):
    """[N,K,D] frozen L2-normed features of K lattice variants (variant 0 = canonical LR)."""
    if cap:
        its = its[:cap]
    pids = np.array([it[1] for it in its]); cams = np.array([it[2] for it in its])
    flat = []
    for it in its:
        hr = ks._to_target_aspect(read_image(it[0]))
        flat.extend(ks.make_lattice_variants(hr, H, K, RNG))     # K PILs
    f = ext.feats_from_pil(flat).reshape(len(its), K, -1)        # [N,K,D]
    return f.astype(np.float32), pids, cams


def per_variant_margin(f, y):
    """margin[n,k] = sim(f_k, true_center) - max_wrong sim(f_k, center).  centers = per-id
    mean of CANONICAL (variant 0) feat.  Returns margin [N,K] and best-variant [N]."""
    uy = np.unique(y)
    cen = np.stack([f[y == c, 0].mean(0) for c in uy]).astype(np.float32)
    cen /= (np.linalg.norm(cen, axis=1, keepdims=True) + 1e-12)
    yi = np.array([{c: i for i, c in enumerate(uy)}[v] for v in y])
    N = len(y)
    sim = f @ cen.T                                              # [N,K,C]
    true_s = sim[np.arange(N)[:, None], np.arange(K)[None, :], yi[:, None]]   # [N,K]
    wrong = sim.copy(); wrong[np.arange(N), :, yi] = -1e9
    margin = true_s - wrong.max(2)                              # [N,K]
    return margin, margin.argmax(1)


# ---- 1. TRAIN: extract variant feats + per-variant margin labels ----
t0 = time.time()
print(f"[LPA] extract TRAIN variant feats (h={H}, K={K}) ...", flush=True)
ft, ytr, _ = variant_feats(items('bounding_box_train'), cap=lpa.train_cap)
N, _, D = ft.shape
margin_tr, best_tr = per_variant_margin(ft, ytr)
print(f"[LPA] train: N={N} D={D}  ({time.time()-t0:.0f}s)", flush=True)

# ---- 2. train the reliability head H(f)->scalar (predict normalized margin) ----
dev = 'cuda'
Xt = torch.tensor(ft.reshape(N * K, D), device=dev)
Mt = torch.tensor(margin_tr.reshape(N * K), device=dev, dtype=torch.float32)
Mt = (Mt - Mt.mean()) / (Mt.std() + 1e-6)
head = torch.nn.Sequential(torch.nn.Linear(D, 256), torch.nn.ReLU(),
                           torch.nn.Linear(256, 1)).to(dev)
opt = torch.optim.Adam(head.parameters(), lr=lpa.lr)
for ep in range(lpa.epochs):
    perm = torch.randperm(N * K, device=dev)
    last = 0.0
    for s in range(0, N * K, 4096):
        idx = perm[s:s + 4096]
        loss = torch.nn.functional.smooth_l1_loss(head(Xt[idx]).squeeze(1), Mt[idx])
        opt.zero_grad(); loss.backward(); opt.step(); last = loss.item()
print(f"[LPA] head trained {lpa.epochs}ep, final loss={last:.4f}", flush=True)

# ---- 3. TEST: weighted vs uniform marginalization ----
print("[LPA] extract QUERY variant feats ...", flush=True)
fq, q_pid, q_cam = variant_feats(items('query'))
if os.path.exists(lpa.cache_gallery):
    gf = np.load(lpa.cache_gallery, allow_pickle=True)['gf']
else:
    gits = items('bounding_box_test')
    gf = ext.feats_from_pil([ks._to_target_aspect(read_image(it[0])) for it in gits])
    np.savez(lpa.cache_gallery, gf=gf)
gits = items('bounding_box_test')
g_pid = np.array([it[1] for it in gits]); g_cam = np.array([it[2] for it in gits])

Xq = torch.tensor(fq.reshape(-1, D), device=dev)
with torch.no_grad():
    a = head(Xq).reshape(len(q_pid), K).cpu().numpy()           # [Nq,K] reliability logits
w = np.exp(a / lpa.tau); w /= w.sum(1, keepdims=True)           # [Nq,K] weights
sim_q = fq @ gf.T                                              # [Nq,K,Ng]
r_uni = ks.eval_map(1 - sim_q.mean(1), q_pid, q_cam, g_pid, g_cam)
r_w = ks.eval_map(1 - (w[:, :, None] * sim_q).sum(1), q_pid, q_cam, g_pid, g_cam)

# diagnostic: does head's argmax match the per-query ORACLE best variant on TEST?
margin_te, best_te = per_variant_margin(fq, q_pid)
acc = float((a.argmax(1) == best_te).mean())
delta = r_w['mAP'] - r_uni['mAP']
print(f"\n[LPA RESULT h={H}]  uniform mAP={r_uni['mAP']:.3f}  weighted mAP={r_w['mAP']:.3f}  "
      f"(weighted-uniform={delta:+.3f})")
print(f"[LPA DIAG] head-argmax == oracle-best-variant acc={100*acc:.1f}%  (chance={100.0/K:.1f}%)")
print(f"[LPA VERDICT] {'PASS (mechanism A lives)' if (delta >= 0.4 and acc >= 0.35) else 'FAIL (best variant not predictable from feature)'}"
      f"  -- need weighted-uniform>=+0.4 AND acc>=35%")
print("[done] LPA head probe complete.")
