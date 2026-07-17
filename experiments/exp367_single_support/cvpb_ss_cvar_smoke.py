#!/usr/bin/env python3
"""Single-Support CVaR frozen-head smoke — codex cheap 路径 #1 (不动 backbone).

冻 backbone, 只训 projection head, 用 episodic single-support CVaR loss。验机制方向:
worst/random-support 是否提升? codex 明确: frozen 失败不判死(可能要改 backbone), 只 smoke。
成功线: worst 或 random-support +0.8~1.0 mAP, full-gallery 掉 <0.5。
对照(防退化 hard-mining): --mode random (episodic CE 无 CVaR), 证不是 episode 本身涨。

训练设计(codex two-level CVaR):
  episode N id × K 图; 每 id 枚举 K 个候选 support, 算该 support 当 prototype 时同 id query 的 CE risk;
  对每 id 的 K 个 support risk 做 CVaR_α(worst tail); L = mean_CE + lam·mean_CVaR。
  (加项不替换主任务; frozen smoke 先只 CE+CVaR, full FT 再加 triplet。)

Run on lab-3090-d: cd repo && python experiments/exp367_single_support/cvpb_ss_cvar_smoke.py
"""
import os, sys, argparse
import numpy as np, torch, torch.nn.functional as F
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth')
ap.add_argument('--config', default='configs/market/pose_psg_lgpa_gcn_base.yml')
ap.add_argument('--eval_cache', default='/tmp/ae_feats.npz')          # query/gallery 特征(复用)
ap.add_argument('--train_cache', default='/tmp/ss_train_feats.npz')
ap.add_argument('--epochs', type=int, default=20)
ap.add_argument('--N', type=int, default=16)        # ids per episode
ap.add_argument('--K', type=int, default=4)         # imgs per id
ap.add_argument('--alpha', type=float, default=0.7) # CVaR tail
ap.add_argument('--lam', type=float, default=0.3)   # CVaR weight
ap.add_argument('--tau', type=float, default=0.1)   # softmax temp
ap.add_argument('--mode', default='cvar', choices=['cvar', 'random'])
cli = ap.parse_args()
sys.argv = ['ks', '--ckpt', cli.ckpt, '--config', cli.config, '--data_root', 'data', '--K', '1',
            '--reuse_gallery', '--cache_gallery', '/tmp/ss_g.npz']
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'experiments', 'cargo_cvpb'))
import cvpb_lattice_killswitch as ks
from datasets.bases import read_image
REPO = ks._repo; DEV = 'cuda'

# ---- 1. 抽 train 特征 (frozen) ----
if os.path.exists(cli.train_cache):
    z = np.load(cli.train_cache); tf, tp = z['tf'], z['tp']
    print(f'[train feat] cached {tf.shape}', flush=True)
else:
    ext = ks.FrozenExtractor()
    its = ks.list_split(os.path.join(REPO, 'data', 'market1501', 'bounding_box_train'))
    pils = [ks._to_target_aspect(read_image(it[0])) for it in its]
    tf = ext.feats_from_pil(pils).astype(np.float32); tf /= np.linalg.norm(tf, axis=1, keepdims=True) + 1e-9
    tp = np.array([it[1] for it in its])
    np.savez(cli.train_cache, tf=tf, tp=tp); print(f'[train feat] extracted {tf.shape}', flush=True)
D = tf.shape[1]
ft = torch.tensor(tf, device=DEV); yt = torch.tensor(tp, device=DEV)
id2idx = defaultdict(list)
for i, p in enumerate(tp): id2idx[int(p)].append(i)
ids = [p for p in id2idx if len(id2idx[p]) >= cli.K]
print(f'[ss-cvar smoke] mode={cli.mode} train-ids={len(ids)} D={D}', flush=True)

# ---- 2. projection head (Linear+BN, init eye) ----
head = torch.nn.Sequential(torch.nn.Linear(D, D, bias=False), torch.nn.BatchNorm1d(D)).to(DEV)
head[0].weight.data.copy_(torch.eye(D))
opt = torch.optim.Adam(head.parameters(), lr=3e-4)


def episode_loss():
    bids = np.random.choice(ids, min(cli.N, len(ids)), replace=False)
    idxs = np.stack([np.random.choice(id2idx[int(y)], cli.K, replace=False) for y in bids])  # [N,K]
    z = F.normalize(head(ft[idxs.reshape(-1)]), dim=1).reshape(len(bids), cli.K, D)            # [N,K,D]
    # 对每个 support-slot s: 用每 id 的第 s 图当 prototype(N 个), 其余 K-1 当 query, 分类到 N prototypes
    id_risks = []                                   # [N] each id 的 CVaR over K support
    ce_terms = []
    for s in range(cli.K):
        proto = z[:, s]                             # [N,D] 每 id 的 support-s 当 prototype
        q_slots = [j for j in range(cli.K) if j != s]
        qz = z[:, q_slots].reshape(len(bids), len(q_slots), D)  # [N,K-1,D]
        logit = torch.einsum('nqd,md->nqm', qz, proto) / cli.tau   # [N,K-1,N] query vs N proto
        tgt = torch.arange(len(bids), device=DEV)[:, None].expand(-1, len(q_slots))  # 正样本=自己 id
        ce = F.cross_entropy(logit.reshape(-1, len(bids)), tgt.reshape(-1), reduction='none').reshape(len(bids), -1)
        ce_terms.append(ce.mean(1))                 # [N] 该 support-s 对每 id 的 risk
    risks = torch.stack(ce_terms, dim=1)            # [N, K] 每 id 的 K 个 support risk
    l_ce = risks.mean()
    if cli.mode == 'random':
        return l_ce                                  # 对照: 无 CVaR
    # CVaR_alpha over K support risks per id (worst tail)
    k_tail = max(1, int(np.ceil((1 - cli.alpha) * cli.K)))
    cvar = torch.topk(risks, k_tail, dim=1).values.mean()
    return l_ce + cli.lam * cvar


for ep in range(cli.epochs):
    head.train(); last = 0.0
    for _ in range(max(1, len(ids) // cli.N)):
        loss = episode_loss()
        opt.zero_grad(); loss.backward(); opt.step(); last = float(loss.item())
    if ep % 5 == 0 or ep == cli.epochs - 1: print(f'  ep{ep} loss={last:.4f}', flush=True)

# ---- 3. 评估: projected query/gallery → full / single-support diagnostic ----
head.eval()
z = np.load(cli.eval_cache)
qf, qp, qc, gf, gp, gc = z['qf'], z['qp'], z['qc'], z['gf'], z['gp'], z['gc']
with torch.no_grad():
    qf = F.normalize(head(torch.tensor(qf, device=DEV)), dim=1).cpu().numpy()
    gf = F.normalize(head(torch.tensor(gf, device=DEV)), dim=1).cpu().numpy()


def eval_fixed(g_idx, valid_q):
    gff, gpp, gcc = gf[g_idx], gp[g_idx], gc[g_idx]
    aps = []
    for i in valid_q:
        s = qf[i] @ gff.T; keep = ~((gpp == qp[i]) & (gcc == qc[i]))
        ss = s[keep]; gpk = gpp[keep]; o = np.argsort(-ss); m = (gpk[o] == qp[i])
        aps.append((np.cumsum(m)[m] / np.arange(1, len(m)+1)[m]).mean() if m.any() else 0.0)
    return 100*np.mean(aps)


id2g = defaultdict(list)
for i, p in enumerate(gp): id2g[p].append(i)
q_ids = set(qp.tolist())
distractor_g = np.array([i for p in id2g if p not in q_ids for i in id2g[p]], dtype=int)
valid_q = np.array([i for i in range(len(qf)) if (gp[~((gp == qp[i]) & (gc == qc[i]))] == qp[i]).any()])


def supp_g(sidx): return np.concatenate([np.array(sidx, dtype=int), distractor_g])
full_mAP = eval_fixed(np.arange(len(gf)), valid_q)
hasq = [p for p in id2g if p in q_ids]
rand_mAPs = []
for sd in range(10):
    rng = np.random.RandomState(sd); rand_mAPs.append(eval_fixed(supp_g([rng.choice(id2g[p]) for p in hasq]), valid_q))
worst_s = []
for p in hasq:
    gi = id2g[p]; qs = np.where(qp == p)[0]
    qual = [(qf[qs[qc[qs] != gc[g]]] @ gf[g]).mean() if (qc[qs] != gc[g]).any() else -1 for g in gi]
    worst_s.append(gi[int(np.argmin(qual))])
worst_mAP = eval_fixed(supp_g(worst_s), valid_q)
print(f'\n[SS-CVAR SMOKE RESULT mode={cli.mode}]')
print(f'  full-gallery   : mAP={full_mAP:.2f}')
print(f'  random-support : mAP={np.mean(rand_mAPs):.2f}±{np.std(rand_mAPs):.2f}')
print(f'  worst-support  : mAP={worst_mAP:.2f}')
print(f'  ※ 与 frozen baseline(probe v2: full 94.43 / random 73.36 / worst 63.82) 比, 看 head 是否抬 worst/random', flush=True)
print('[done]', flush=True)
