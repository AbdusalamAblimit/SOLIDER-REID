#!/usr/bin/env python3
"""Active Evidence Acquisition ReID — cheap kill-switch (零训练).

codex 范式级方向 #1 (7/10): query 不只被动排序, 系统可花预算主动获取视觉证据(另一 camera 视角)。
★真 kill-switch(非 codex 的 trivial oracle, multi-query 必涨): policy(hard query 选预算)能否接近 oracle?

  - baseline      : single query mAP
  - oracle-all    : 每 query + 同 ID 不同 camera 第二证据(multi-query mean) → upper-bound
  - **policy**    : 只对 hard query(top-1 margin 小=不确定) 花预算 budget% 获取第二证据
  - random        : 随机 budget% query 给第二证据 (对照)

GO: policy gain / oracle-all gain >= 0.5  AND  policy 明显 > random → 主动获取证据 policy 有真价值。
DEAD: policy ≈ random → 没 policy 价值(等于 trivial multi-query, 预算分配无效)。
控 margin(top1-top2 sim) = #false-in-topk 的代理。frozen SOLIDER, 零训练。

Run on lab-3090-d:
  cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 /root/miniconda3/envs/solider-reid/bin/python \
    experiments/exp366_active_evidence/cvpb_active_evidence_probe.py \
    --ckpt log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth 2>&1 | tee /tmp/cvpb_ae.log
"""
import sys, os, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth')
ap.add_argument('--config', default='configs/market/pose_psg_lgpa_gcn_base.yml')
ap.add_argument('--budget', type=float, default=0.2)       # 20% query 可获取第二证据
ap.add_argument('--data_dir', default='market1501')        # market1501 / occluded_duke
ap.add_argument('--cache', default='/tmp/ae_feats.npz')
cli = ap.parse_args()
sys.argv = ['ks', '--ckpt', cli.ckpt, '--config', cli.config, '--data_root', 'data', '--K', '1',
            '--reuse_gallery', '--cache_gallery', '/tmp/ae_g.npz']
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'experiments', 'cargo_cvpb'))
import cvpb_lattice_killswitch as ks
from datasets.bases import read_image

REPO = ks._repo; ext = ks.FrozenExtractor()


def extract(split):
    items = ks.list_split(os.path.join(REPO, 'data', cli.data_dir, split))
    pils = [ks._to_target_aspect(read_image(it[0])) for it in items]
    feats = ext.feats_from_pil(pils).astype(np.float32)
    feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    pid = np.array([it[1] for it in items]); cam = np.array([it[2] for it in items])
    return feats, pid, cam


if os.path.exists(cli.cache):
    z = np.load(cli.cache)
    qf, qp, qc, gf, gp, gc = z['qf'], z['qp'], z['qc'], z['gf'], z['gp'], z['gc']
    print('[feat] cached', flush=True)
else:
    print('[AE] extract query/gallery feats ...', flush=True)
    qf, qp, qc = extract('query')
    gf, gp, gc = extract('bounding_box_test')
    np.savez(cli.cache, qf=qf, qp=qp, qc=qc, gf=gf, gp=gp, gc=gc)
print(f'[AE] q={len(qf)} g={len(gf)}', flush=True)


def eval_market(qfeat, qp, qc, gf, gp, gc):
    """标准 Market mAP/R1: 排除同 camera 同 ID gallery。返回 mAP + per-query margin(top1-top2)。"""
    sim = qfeat @ gf.T
    aps, r1s = [], []
    margins = np.ones(len(qfeat))                        # 每 query 都算(难度, 不依赖 match)
    for i in range(len(qfeat)):
        keep = ~((gp == qp[i]) & (gc == qc[i]))         # 排同 cam 同 id
        s = sim[i][keep]; gp_k = gp[keep]
        ss = np.sort(s)[::-1]
        margins[i] = float(ss[0] - ss[1]) if len(ss) > 1 else 1.0
        o = np.argsort(-s); gp_o = gp_k[o]; match = (gp_o == qp[i])
        if not match.any(): continue
        cum = np.cumsum(match); ranks = np.arange(1, len(gp_o) + 1)
        aps.append((cum[match] / ranks[match]).mean()); r1s.append(float(match[0]))
    return 100*np.mean(aps), 100*np.mean(r1s), margins


# 每 query 的第二证据 = 同 ID 不同 camera 的另一张 query 图 (无则无证据)
from collections import defaultdict
idc2q = defaultdict(list)
for i in range(len(qf)): idc2q[(qp[i], qc[i])].append(i)
second = -np.ones(len(qf), dtype=int)
for i in range(len(qf)):
    cands = [j for j in range(len(qf)) if qp[j] == qp[i] and qc[j] != qc[i]]
    if cands: second[i] = cands[0]
has_second = second >= 0
print(f'[AE] queries with 2nd-evidence available: {has_second.sum()}/{len(qf)}', flush=True)


def with_evidence(use_mask):
    """对 use_mask 的 query 用 (q + 2nd)/2 multi-query, 其余 single。"""
    qq = qf.copy()
    for i in np.where(use_mask & has_second)[0]:
        qq[i] = (qf[i] + qf[second[i]]); qq[i] /= (np.linalg.norm(qq[i]) + 1e-9)
    return qq


# baseline single
base_mAP, base_R1, margins = eval_market(qf, qp, qc, gf, gp, gc)
# oracle-all: 所有有 2nd 的 query 都用
orc_mAP, orc_R1, _ = eval_market(with_evidence(np.ones(len(qf), bool)), qp, qc, gf, gp, gc)
# policy: margin 最小的 budget% (hard) 用证据
n_budget = int(cli.budget * has_second.sum())
# policy: 只在"有第二证据可获取"的 query 里, 选 margin 最小(hard)的 budget 个 (和 random 同池公平对照)
cand = np.where(has_second)[0]
hard = np.zeros(len(qf), bool); hard[cand[np.argsort(margins[cand])[:n_budget]]] = True
pol_mAP, pol_R1, _ = eval_market(with_evidence(hard), qp, qc, gf, gp, gc)
# random: 随机 budget%
rng = np.random.RandomState(42)
rmask = np.zeros(len(qf), bool); ridx = rng.choice(np.where(has_second)[0], min(n_budget, has_second.sum()), replace=False); rmask[ridx] = True
rnd_mAP, rnd_R1, _ = eval_market(with_evidence(rmask), qp, qc, gf, gp, gc)

print(f'\n[AE RESULT] budget={cli.budget}')
print(f'  baseline single   : mAP={base_mAP:.2f} R1={base_R1:.2f}')
print(f'  oracle-all (2nd)  : mAP={orc_mAP:.2f} R1={orc_R1:.2f}  gain={orc_mAP-base_mAP:+.2f}')
print(f'  policy (hard {cli.budget:.0%}): mAP={pol_mAP:.2f} R1={pol_R1:.2f}  gain={pol_mAP-base_mAP:+.2f}')
print(f'  random ({cli.budget:.0%})     : mAP={rnd_mAP:.2f} R1={rnd_R1:.2f}  gain={rnd_mAP-base_mAP:+.2f}')
orc_gain = orc_mAP - base_mAP; pol_gain = pol_mAP - base_mAP; rnd_gain = rnd_mAP - base_mAP
frac = pol_gain / orc_gain if orc_gain > 0.1 else 0.0
print(f'  policy/oracle gain frac = {frac:.2f} (GO>=0.5)  policy-random = {pol_gain-rnd_gain:+.2f} (GO>0)')
go = frac >= 0.5 and (pol_gain - rnd_gain) > 0.3
print(f'  [verdict] {"GO (active evidence policy 有价值)" if go else "DEAD (policy≈random 或 oracle 本身弱, 主动获取无 policy 价值)"}')
print('[done]', flush=True)
