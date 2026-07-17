#!/usr/bin/env python3
"""Wildlife local-verifier cheap kill-switch (codex 建议最后一搏, 零训练).

GiraffeZebraID baseline(frozen MegaDescriptor): false_top10=0.712 全 same-species hard neg。
验: LoFTR local matching(query vs top-k gallery)能否纠正 same-species false?
  压 false_top10 < 0.60  AND  R1 +>=1  → pattern-token 蒸馏有戏(虽撞 WildFusion 偏弱);
  压不下 → 彻底停 Wildlife, 收 LM-ReID 6.5。
复用已抽 MegaDescriptor 特征 + 采样 query(LoFTR 慢)。

Run: cd experiments/exp365_wildlife && .venv-wl/bin/python cvpb_wildlife_localverify.py
"""
import os, argparse, numpy as np, torch
from PIL import Image
import kornia.feature as KF
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='data/species/GiraffeZebraID')
ap.add_argument('--name', default='GiraffeZebraID')
ap.add_argument('--cache', default='/tmp/wl_feats_gz.npz')
ap.add_argument('--n_query', type=int, default=200)   # 采样 query (LoFTR 慢)
ap.add_argument('--topk', type=int, default=10)
ap.add_argument('--sz', type=int, default=480)
cli = ap.parse_args()

from wildlife_datasets import datasets as wd
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
ds = getattr(wd, cli.name)(cli.data); df = ds.df.reset_index(drop=True); root = ds.root
feats = np.load(cli.cache)['feats']
pid = df.identity.values.astype(str); species = df.species.values.astype(str)

id2idx = defaultdict(list)
for i, p in enumerate(pid): id2idx[p].append(i)
q_idx, g_idx = [], []
for p, idxs in id2idx.items():
    if len(idxs) >= 2: q_idx.append(idxs[0]); g_idx.extend(idxs[1:])
q_idx = np.array(q_idx); g_idx = np.array(g_idx)
qf, gf = feats[q_idx], feats[g_idx]; qp, gp = pid[q_idx], pid[g_idx]
sim = qf @ gf.T; order = np.argsort(-sim, axis=1)

matcher = KF.LoFTR(pretrained='outdoor').eval().to(DEV)


def load_gray(idx):
    r = df.iloc[idx]; img = Image.open(os.path.join(root, r['path'])).convert('L')
    b = r.get('bbox')
    if isinstance(b, (list, tuple, np.ndarray)) and len(b) == 4:
        x, y, w, h = [float(v) for v in b]
        if w > 1 and h > 1: img = img.crop((x, y, x + w, y + h))
    img = img.resize((cli.sz, cli.sz))
    return torch.tensor(np.array(img), dtype=torch.float32)[None, None] / 255.


def n_match(qi, gi):
    with torch.no_grad():
        out = matcher({'image0': load_gray(qi).to(DEV), 'image1': load_gray(gi).to(DEV)})
        return float((out['confidence'] > 0.5).sum())


rng = np.random.RandomState(42)
sel = rng.choice(len(q_idx), min(cli.n_query, len(q_idx)), replace=False)
b_false, b_r1, lv_false, lv_r1 = [], [], [], []
for c, ql in enumerate(sel):
    o = order[ql][:cli.topk]; gp_tk = gp[o]; q = qp[ql]
    if (gp_tk == q).sum() == 0:     # top-k 内无正样本, rerank 无意义, 跳过(对两者公平)
        continue
    b_false.append((gp_tk != q).mean()); b_r1.append(float(gp_tk[0] == q))
    nm = np.array([n_match(q_idx[ql], g_idx[gi]) for gi in o])
    gp_lv = gp_tk[np.argsort(-nm)]
    lv_false.append((gp_lv != q).mean()); lv_r1.append(float(gp_lv[0] == q))
    if c % 50 == 0: print(f'  {c}/{len(sel)}', flush=True)

print(f'\n[local-verifier] n_eval={len(b_r1)} topk={cli.topk}')
print(f'  baseline    : false_top{cli.topk}={np.mean(b_false):.3f}  R1(in-topk)={100*np.mean(b_r1):.2f}')
print(f'  LoFTR rerank: false_top{cli.topk}={np.mean(lv_false):.3f}  R1(in-topk)={100*np.mean(lv_r1):.2f}')
print(f'  Δfalse={np.mean(lv_false)-np.mean(b_false):+.3f}  ΔR1={100*(np.mean(lv_r1)-np.mean(b_r1)):+.2f}')
go = (np.mean(lv_false) < 0.60) and (100*(np.mean(lv_r1)-np.mean(b_r1)) >= 1.0)
print(f'  [verdict] {"GO (local-verifier 纠正 same-species hard neg)" if go else "DEAD (local matching 救不了 → 彻底停 Wildlife, 收 LM-ReID 6.5)"}')
print('[done]', flush=True)
