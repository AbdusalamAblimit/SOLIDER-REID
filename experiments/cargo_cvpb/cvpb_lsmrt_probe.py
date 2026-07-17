#!/usr/bin/env python3
"""LS-MRT probe -- train-time mechanism (Lattice-Set Marginal Retrieval Training), frozen backbone.

Train a small linear re-projection P with a SET-RETRIEVAL SupCon loss that is ISOMORPHIC to the
test-time decision marginalization:  S(q,g) = logmeanexp_k sim(P z_{q,k}, P z_g).  Marginalize on
the RETRIEVAL decision layer (q-g similarity, real negatives in denominator) -- NOT on a train-ID
classifier head (that was L_marg, which collapsed).  Frozen backbone + cached feats => cheap probe.

PASS if h=16 set-score-with-P  >=  uniform-lattice-marg + 0.3 mAP  AND  K-variant mean cosine does
NOT rise (no collapse).  Train/test ids DISJOINT (no leakage)."""
import sys, os, numpy as np, argparse
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='log/market1501/exp359_abl_noLMloss/transformer_40.pth')
ap.add_argument('--config', default='configs/market/pose_psg_lgpa_gcn_base.yml')
ap.add_argument('--h', type=int, default=16)
ap.add_argument('--K', type=int, default=9)
ap.add_argument('--epochs', type=int, default=30)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--tau_l', type=float, default=0.1)     # lattice marginalization temp
ap.add_argument('--tau_c', type=float, default=0.1)     # contrastive temp
ap.add_argument('--P', type=int, default=16)            # ids per batch
ap.add_argument('--Kins', type=int, default=4)          # instances per id
ap.add_argument('--train_cap', type=int, default=0)
ap.add_argument('--lambda_id', type=float, default=0.1, help='identity reg ||P-I||^2 to keep P near identity (prevent overfit; smoke showed full linear P overfits)')
ap.add_argument('--cache_gallery', default='/tmp/g_lpa.npz')
cli = ap.parse_args()
sys.argv = ['ks', '--ckpt', cli.ckpt, '--config', cli.config, '--data_root', 'data',
            '--K', str(cli.K), '--reuse_gallery', '--cache_gallery', cli.cache_gallery]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvpb_lattice_killswitch as ks
import torch, torch.nn.functional as F
from datasets.bases import read_image

RNG = np.random.RandomState(42); H, K = cli.h, cli.K
ext = ks.FrozenExtractor(); REPO = ks._repo
DEV = 'cuda'


def items(s):
    return ks.list_split(os.path.join(REPO, 'data', 'market1501', s))


def variant_feats(its, cap=0):
    if cap:
        its = its[:cap]
    pids = np.array([it[1] for it in its]); cams = np.array([it[2] for it in its])
    flat = []
    for it in its:
        hr = ks._to_target_aspect(read_image(it[0]))
        flat.extend(ks.make_lattice_variants(hr, H, K, RNG))
    return ext.feats_from_pil(flat).reshape(len(its), K, -1).astype(np.float32), pids, cams


def setscore(zq, zg, tau_l):
    """logmeanexp_k sim(zq_{i,k}, zg_j) -> [Nq,Ng].  zq [Nq,K,D] L2, zg [Ng,D] L2."""
    sim = np.einsum('ikd,jd->ijk', zq, zg)                      # [Nq,Ng,K] cos
    return tau_l * (np.log(np.exp(sim / tau_l).mean(2) + 1e-12))  # logmeanexp over K


# ---- 1. TRAIN: cache variant feats, fit P with set-retrieval SupCon ----
print(f"[LS-MRT] extract TRAIN variant feats (h={H}, K={K}) ...", flush=True)
ft, ytr, _ = variant_feats(items('bounding_box_train'), cap=cli.train_cap)
N, _, D = ft.shape
ft_t = torch.tensor(ft, device=DEV); yt = torch.tensor(ytr, device=DEV)
id2idx = defaultdict(list)
for i, y in enumerate(ytr):
    id2idx[int(y)].append(i)
ids = [y for y in id2idx if len(id2idx[y]) >= 2]
P = torch.nn.Linear(D, D, bias=False).to(DEV)
P.weight.data.copy_(torch.eye(D, device=DEV))               # init = identity
opt = torch.optim.Adam(P.parameters(), lr=cli.lr)
iters = max(1, N // (cli.P * cli.Kins))
for ep in range(cli.epochs):
    last = 0.0
    for _ in range(iters):
        bids = RNG.choice(ids, min(cli.P, len(ids)), replace=False)
        bidx = []
        for y in bids:
            pool = id2idx[int(y)]
            bidx.extend(RNG.choice(pool, cli.Kins, replace=len(pool) < cli.Kins))
        bidx = np.array(bidx)
        zb = F.normalize(P(ft_t[bidx]), dim=-1)                # [b,K,D]
        zg = zb[:, 0]                                          # [b,D] canonical-0 as in-batch gallery
        sim = torch.einsum('ikd,jd->ijk', zb, zg) / cli.tau_l  # [b,b,K]
        S = cli.tau_l * torch.logsumexp(sim - float(np.log(K)), dim=2)   # [b,b] logmeanexp_k
        yb = yt[bidx]
        pos = (yb[:, None] == yb[None, :]).float(); pos.fill_diagonal_(0)
        logits = S / cli.tau_c; logits.fill_diagonal_(-1e9)
        logp = F.log_softmax(logits, dim=1)
        loss = (-(pos * logp).sum(1) / pos.sum(1).clamp_min(1)).mean()
        loss = loss + cli.lambda_id * ((P.weight - torch.eye(D, device=DEV)) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step(); last = loss.item()
print(f"[LS-MRT] P fit {cli.epochs}ep, final loss={last:.4f}", flush=True)

# ---- 2. TEST: set-score with P vs uniform (P=I) ----
print("[LS-MRT] extract QUERY variant feats ...", flush=True)
fq, q_pid, q_cam = variant_feats(items('query'))
gf = np.load(cli.cache_gallery, allow_pickle=True)['gf']
gits = items('bounding_box_test')
g_pid = np.array([it[1] for it in gits]); g_cam = np.array([it[2] for it in gits])
with torch.no_grad():
    zq = F.normalize(P(torch.tensor(fq, device=DEV)), dim=-1).cpu().numpy()      # [Nq,K,D]
    zg = F.normalize(P(torch.tensor(gf, device=DEV)), dim=-1).cpu().numpy()      # [Ng,D]
uq = fq / (np.linalg.norm(fq, axis=2, keepdims=True) + 1e-12)                    # uniform: no P
ug = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
r_P = ks.eval_map(-setscore(zq, zg, cli.tau_l), q_pid, q_cam, g_pid, g_cam)
r_U = ks.eval_map(-setscore(uq, ug, cli.tau_l), q_pid, q_cam, g_pid, g_cam)
# collapse check: mean pairwise cosine among the K variants (test queries), with vs without P
cos_P = float(np.mean([(zq[i] @ zq[i].T)[np.triu_indices(K, 1)].mean() for i in range(len(q_pid))]))
cos_U = float(np.mean([(uq[i] @ uq[i].T)[np.triu_indices(K, 1)].mean() for i in range(len(q_pid))]))
delta = r_P['mAP'] - r_U['mAP']
print(f"\n[LS-MRT RESULT h={H}]  uniform-setmarg mAP={r_U['mAP']:.3f}  P-setmarg mAP={r_P['mAP']:.3f}  "
      f"(P-uniform={delta:+.3f})")
print(f"[LS-MRT DIAG] K-variant mean cos: uniform={cos_U:.4f}  withP={cos_P:.4f}  "
      f"(rise={cos_P-cos_U:+.4f}; rise>0 => collapse risk)")
print(f"[LS-MRT VERDICT] {'PASS (set-retrieval training helps)' if (delta >= 0.3 and cos_P - cos_U <= 0.02) else 'FAIL'}"
      f"  -- need P-uniform>=+0.3 AND no cosine collapse")
print("[done] LS-MRT probe complete.")
