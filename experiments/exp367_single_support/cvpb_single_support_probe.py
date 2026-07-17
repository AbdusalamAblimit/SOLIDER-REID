#!/usr/bin/env python3
"""Single-Support ReID — cheap kill-switch (零训练) — v2 (codex needs-attention 修).

codex 训练侧 #1: 训练时每 ID 单图 support 定义身份, CVaR worst-support 优化。回应 exp109
根问题(single-image support incomplete)。纯训练侧(episodic loss 输出常规 descriptor)。

★codex 审 v1 抓 3 个 High, v2 修:
  1. common-valid query mask: 用 full-gallery 下有 positive 的 query 子集, 所有 support 设置同子集(否则比不同难度)。
  2. distractor/no-query pid 保持全量(不压 1 张), 只对 has-query ID 做 single-support(负样本池不变, #false-in-topk 可比)。
  3. 主判据 best-random / random-worst 多 seed(20) 均值±std; 报 #false-in-topk(top10 错样本数)。
  4. best/worst 用 query-label oracle(诊断上下界, 不证训练可学, 诚实标注)。
  5. cache provenance: 校验 full-gallery mAP sanity(=exp260b ref 94.4)。

GO(support 选择是真训练瓶颈): random-worst gap > 3 mAP(多 seed 稳, 同负样本池同 valid query) AND
   #false-in-topk 不被 trivial 解释。DEAD: gap 小或被负样本池/valid-query 变化解释。

Run on lab-3090-d: cd repo && python experiments/exp367_single_support/cvpb_single_support_probe.py
"""
import numpy as np
from collections import defaultdict
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--cache', default='/tmp/ae_feats.npz')
ap.add_argument('--seeds', type=int, default=20)
cli = ap.parse_args()

import os
if not os.path.exists(cli.cache):
    raise SystemExit(f'[FATAL] cache {cli.cache} 不存在, 先跑 active_evidence probe 生成 Market 特征 cache')
z = np.load(cli.cache)
qf, qp, qc = z['qf'], z['qp'], z['qc']
gf, gp, gc = z['gf'], z['gp'], z['gc']
assert np.isfinite(qf).all() and np.isfinite(gf).all(), 'feat 含 nan/inf'
print(f'[single-support v2] q={len(qf)} g={len(gf)} g-ids={len(set(gp))}', flush=True)

# pid 分类: has-query ID(在 query 出现) vs distractor(只在 gallery)
q_ids = set(qp.tolist())
id2g = defaultdict(list)
for i, p in enumerate(gp): id2g[p].append(i)
hasq_ids = [p for p in id2g if p in q_ids]
distractor_g = np.array([i for p in id2g if p not in q_ids for i in id2g[p]], dtype=int)
print(f'[single-support v2] has-query ids={len(hasq_ids)} distractor-gallery imgs={len(distractor_g)}', flush=True)


def support_gallery(support_idx_per_id):
    """has-query ID 用单 support, distractor 全量 → 负样本池不变。"""
    return np.concatenate([np.array(support_idx_per_id, dtype=int), distractor_g])


def eval_fixed(g_idx, valid_q):
    """对固定 valid_q 子集 eval mAP/R1 + #false-in-top10。g_idx=gallery 子集。"""
    gff, gpp, gcc = gf[g_idx], gp[g_idx], gc[g_idx]
    aps, r1s, false10 = [], [], []
    for i in valid_q:
        sim_i = qf[i] @ gff.T
        keep = ~((gpp == qp[i]) & (gcc == qc[i]))
        s = sim_i[keep]; gpk = gpp[keep]
        o = np.argsort(-s); m = (gpk[o] == qp[i])
        if not m.any():
            aps.append(0.0); r1s.append(0.0); false10.append(1.0); continue   # missing-positive 记 0(codex)
        cum = np.cumsum(m); r = np.arange(1, len(m) + 1)
        aps.append((cum[m] / r[m]).mean()); r1s.append(float(m[0]))
        false10.append(float((gpk[o[:10]] != qp[i]).mean()))
    return 100*np.mean(aps), 100*np.mean(r1s), np.mean(false10)


# common-valid query: full-gallery 下有 positive 的 query (固定子集, 所有 support 设置共用)
full_g = np.arange(len(gf))
valid_q = []
for i in range(len(qf)):
    keep = ~((gp == qp[i]) & (gc == qc[i]))
    if (gp[keep] == qp[i]).any(): valid_q.append(i)
valid_q = np.array(valid_q)
print(f'[single-support v2] common-valid query={len(valid_q)}/{len(qf)}', flush=True)

full_mAP, full_R1, full_f10 = eval_fixed(full_g, valid_q)
print(f'  full-gallery sanity mAP={full_mAP:.2f} (provenance check vs exp260b ref ~94.4)', flush=True)

# best/worst-support oracle (用 query-label, 诊断上下界 — 诚实: 不证训练可学)
best_s, worst_s = [], []
for p in hasq_ids:
    gidxs = id2g[p]; q_same = np.where(qp == p)[0]
    qual = []
    for g in gidxs:
        qs = q_same[qc[q_same] != gc[g]]
        qual.append((qf[qs] @ gf[g]).mean() if len(qs) else -1.0)
    qual = np.array(qual)
    best_s.append(gidxs[int(np.argmax(qual))]); worst_s.append(gidxs[int(np.argmin(qual))])
best_mAP, best_R1, best_f10 = eval_fixed(support_gallery(best_s), valid_q)
worst_mAP, worst_R1, worst_f10 = eval_fixed(support_gallery(worst_s), valid_q)

# random-support 多 seed
rand_mAPs = []
for sd in range(cli.seeds):
    rng = np.random.RandomState(sd)
    rs = [rng.choice(id2g[p]) for p in hasq_ids]
    rand_mAPs.append(eval_fixed(support_gallery(rs), valid_q)[0])
rand_mean, rand_std = np.mean(rand_mAPs), np.std(rand_mAPs)

print(f'\n[SINGLE-SUPPORT v2 RESULT] (common-valid q={len(valid_q)}, distractor 全量, {cli.seeds} seeds)')
print(f'  full-gallery   : mAP={full_mAP:.2f} R1={full_R1:.2f} false10={full_f10:.3f}')
print(f'  best-support   : mAP={best_mAP:.2f} (oracle 上界, 用 query-label)  false10={best_f10:.3f}')
print(f'  random-support : mAP={rand_mean:.2f}±{rand_std:.2f}')
print(f'  worst-support  : mAP={worst_mAP:.2f} (oracle 下界)  false10={worst_f10:.3f}')
print(f'  best-random gap = {best_mAP-rand_mean:.2f}  random-worst gap = {rand_mean-worst_mAP:.2f}  best-worst = {best_mAP-worst_mAP:.2f}')
# 主判据: random-worst gap(同负样本池同valid query, 单support内选择) > 3 且 false10 同向变化(非trivial少正样本)
go = (rand_mean - worst_mAP) > 3.0 and (best_mAP - rand_mean) > 1.0
print(f'  [verdict] {"GO (单 support 内选择 matters, support representation 是真训练瓶颈)" if go else "DEAD (support 选择价值小/被负样本池-valid-query 解释)"}')
print('  ※ best/worst 是 query-label oracle 诊断上下界, 证 headroom 存在; 训练能否学到要 CVaR train 验', flush=True)
print('[done]', flush=True)
