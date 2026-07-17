#!/usr/bin/env python3
"""Wildlife species-conditioned frozen probe — cheap kill-switch (零训练).

exp365 design + codex_wildlife_check kill-switch。frozen MegaDescriptor-L-384 抽特征,
验"多物种少样本下 species-conditioned metric 有空间吗"(non-WildlifeReID10k 用 lila 多物种拼)。

测量:
  - baseline mAP (all-species gallery)
  - **per-species centering oracle**: 每 species 减 species-mean(白化) → mAP (species-conditioned 空间?)
  - all-species gallery vs same-species-only gallery (species 干扰存在吗)
  - 指标: mAP + Rank-1 + false_in_top10 + **wrong-species-in-top10** (错误是否 cross-species)

Go: per-species centering oracle >= +3 mAP  AND  all-species 明显比 same-species 差  AND
    错误集中 same-species hard-neg(wrong-species 低=同物种内难).
Kill: per-species centering < +1 (MegaDescriptor 已 species-agnostic 强, 无 conditioning 空间)
    OR 错误主要 cross-species(说明 backbone 连物种都分不清, 不是 ReID 问题).

Run (本地 py3.11 .venv-wl, MPS):
  cd experiments/exp365_wildlife && .venv-wl/bin/python cvpb_wildlife_probe.py --name GiraffeZebraID
"""
import os, argparse
import numpy as np
import torch, timm
from PIL import Image
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='data/species/GiraffeZebraID')
ap.add_argument('--name', default='GiraffeZebraID')
ap.add_argument('--model', default='hf-hub:BVRA/MegaDescriptor-L-384')
ap.add_argument('--batch', type=int, default=32)
ap.add_argument('--cache', default='/tmp/wl_feats_gz.npz')
cli = ap.parse_args()

from wildlife_datasets import datasets as wd
DEV = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

ds = getattr(wd, cli.name)(cli.data); df = ds.df.reset_index(drop=True); root = ds.root
pid = df.identity.values.astype(str); species = df.species.values.astype(str)
print(f'[{cli.name}] {len(df)} imgs, {len(set(pid))} ids, species {df.species.value_counts().to_dict()}', flush=True)

# ---- 1. MegaDescriptor 抽特征 (bbox crop) ----
if os.path.exists(cli.cache):
    feats = np.load(cli.cache)['feats']
    print(f'[feat] cached {feats.shape}', flush=True)
else:
    model = timm.create_model(cli.model, pretrained=True, num_classes=0).eval().to(DEV)
    cfg = timm.data.resolve_data_config({}, model=model); tf = timm.data.create_transform(**cfg)
    feats = []
    for i in range(0, len(df), cli.batch):
        batch = []
        for _, r in df.iloc[i:i+cli.batch].iterrows():
            img = Image.open(os.path.join(root, r['path'])).convert('RGB')
            b = r.get('bbox')
            if isinstance(b, (list, tuple, np.ndarray)) and len(b) == 4:
                x, y, w, h = [float(v) for v in b]
                if w > 1 and h > 1: img = img.crop((x, y, x + w, y + h))
            batch.append(tf(img))
        with torch.no_grad():
            f = model(torch.stack(batch).to(DEV))
        feats.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
        if i % 512 == 0: print(f'  feat {i}/{len(df)}', flush=True)
    feats = np.concatenate(feats).astype(np.float32)
    np.savez(cli.cache, feats=feats)
    print(f'[feat] extracted {feats.shape}', flush=True)

# ---- 2. split: 每 identity 第1张 query, 其余 gallery (只用 >=2 图的 id) ----
id2idx = defaultdict(list)
for i, p in enumerate(pid): id2idx[p].append(i)
q_idx, g_idx = [], []
for p, idxs in id2idx.items():
    if len(idxs) >= 2:
        q_idx.append(idxs[0]); g_idx.extend(idxs[1:])
q_idx = np.array(q_idx); g_idx = np.array(g_idx)
print(f'[split] q={len(q_idx)} g={len(g_idx)}', flush=True)


def eval_map(qf, gf, qp, gp, qs, gs, topk=10):
    """标准 ReID mAP/R1 + false_in_topk + wrong-species-in-topk。"""
    sim = qf @ gf.T                                  # [Nq,Ng] cos
    order = np.argsort(-sim, axis=1)
    aps, r1s, false_tk, wrongsp_tk = [], [], [], []
    for i in range(len(qf)):
        o = order[i]; match = (gp[o] == qp[i])
        if not match.any(): continue
        # AP
        cum = np.cumsum(match); ranks = np.arange(1, len(o) + 1)
        ap = (cum[match] / ranks[match]).mean(); aps.append(ap)
        r1s.append(float(match[0]))
        tk = o[:topk]
        false_tk.append(float((gp[tk] != qp[i]).mean()))       # #false-in-topk (trivial 代理)
        wrongsp_tk.append(float((gs[tk] != qs[i]).mean()))     # wrong-species-in-topk
    return dict(mAP=100*np.mean(aps), R1=100*np.mean(r1s),
                false_top=np.mean(false_tk), wrongsp_top=np.mean(wrongsp_tk))


qf, gf = feats[q_idx], feats[g_idx]
qp, gp = pid[q_idx], pid[g_idx]
qs, gs = species[q_idx], species[g_idx]

# (A) baseline all-species
base = eval_map(qf, gf, qp, gp, qs, gs)
print(f'\n[A baseline all-species] mAP={base["mAP"]:.2f} R1={base["R1"]:.2f} '
      f'false_top10={base["false_top"]:.3f} wrong-species_top10={base["wrongsp_top"]:.3f}', flush=True)

# (B) per-species centering oracle: 每 species 减该 species 全局 mean 再 L2
def species_center(f, sp):
    out = f.copy()
    for s in set(sp):
        m = f[sp == s].mean(0)
        out[sp == s] = f[sp == s] - m
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)

qf_c = species_center(feats, species)[q_idx]; gf_c = species_center(feats, species)[g_idx]
cen = eval_map(qf_c, gf_c, qp, gp, qs, gs)
print(f'[B per-species centering oracle] mAP={cen["mAP"]:.2f} R1={cen["R1"]:.2f}  '
      f'gain={cen["mAP"]-base["mAP"]:+.2f} (Go>=+3)', flush=True)

# (C) same-species-only gallery (每 query 只检索同物种 gallery)
aps_same = []
sim = qf @ gf.T
for i in range(len(qf)):
    same = (gs == qs[i])
    if not same.any(): continue
    o = np.argsort(-sim[i][same]); gp_s = gp[same][o]; match = (gp_s == qp[i])
    if not match.any(): continue
    cum = np.cumsum(match); ranks = np.arange(1, len(gp_s) + 1)
    aps_same.append((cum[match] / ranks[match]).mean())
mAP_same = 100 * np.mean(aps_same)
print(f'[C same-species-only gallery] mAP={mAP_same:.2f}  '
      f'(all-species {base["mAP"]:.2f}; species 干扰={mAP_same-base["mAP"]:+.2f})', flush=True)

# ---- verdict ----
go_center = (cen["mAP"] - base["mAP"]) >= 3.0
go_interf = (mAP_same - base["mAP"]) >= 1.0
print(f'\n[VERDICT] per-species-centering gain={cen["mAP"]-base["mAP"]:+.2f}(Go>=+3:{go_center}) '
      f'species-interference={mAP_same-base["mAP"]:+.2f}(Go>=+1:{go_interf})', flush=True)
print('  GO (species-conditioned 有空间)' if (go_center or go_interf) else
      '  DEAD (MegaDescriptor 已 species-agnostic 强, 无 conditioning 空间)', flush=True)
print('[done]', flush=True)
