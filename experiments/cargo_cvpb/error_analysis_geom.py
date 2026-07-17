#!/usr/bin/env python3
"""① Geometry-Aware Ambiguity 零训练 kill-switch.

冻结 AG-ReID.v2 baseline(swin_small SOLIDER),提 BN 特征 → cosine distmat,
统计 baseline 错误是否集中在"几何歧义样本"(高 altitude + 大 scale 差):
  (a) 按 query altitude(文件夹名 A0/A1/A2)分桶 mAP —— 高度↑是否难↑
  (b) top-1 假阳: 错配 gallery 与 query 的 scale 比是否接近 1(几何相似负样本)
  (c) 真值 rank: 真匹配 scale 差是否大(几何不相似正样本被推远)
  (d) 几何歧义子集(高 altitude OR 大 scale 差) vs 全集 mAP gap

在 lab-4090 跑(GPU 空):
  cd /home/afr/SOLIDER-REID/experiments/cargo_cvpb && \
  python error_analysis_geom.py --ckpt /home/afr/SOLIDER-REID/log/cargo/baseline_agreidv2_4090/model_final.pth \
    --data_root /home/afr/SOLIDER-REID/data \
    --swin_pretrain /home/afr/SOLIDER-REID/pretrained/swin_small.pth 2>&1 | tee /tmp/err_analysis_geom.log
"""
import os, re, sys, argparse
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)                                    # cargo_cvpb: afd_train
sys.path.insert(0, os.path.join(_here, '..', 'afd_reid'))    # afd_reid: afd_model, agreid_v2_combined, cargo_dataset
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--data_root', default='/home/afr/SOLIDER-REID/data')
ap.add_argument('--swin_pretrain', default='/home/afr/SOLIDER-REID/pretrained/swin_small.pth')
ap.add_argument('--img_size', type=int, nargs=2, default=[256, 128])
ap.add_argument('--workers', type=int, default=4)
cli = ap.parse_args()

# minimal namespace for build_model (uses getattr defaults) + build_eval_loader
args = argparse.Namespace(
    backbone='swin_small', swin_pretrain=cli.swin_pretrain, swin_semantic_weight=0.2,
    img_size=cli.img_size, use_afd=False, dataset='agreid_v2', data_root=cli.data_root,
    workers=cli.workers, test_batch=64, ovli=False, ovp=False,
)
device = 'cuda'

from afd_model import build_model
from agreid_v2_combined import AGReIDV2Combined
try:
    from cargo_dataset import filter_by_view
except Exception:
    from afd_train import filter_by_view
from afd_train import build_eval_loader

ds = AGReIDV2Combined(root=args.data_root, verbose=True)
model = build_model(ds.num_train_pids, args).to(device)
ck = torch.load(cli.ckpt, map_location='cpu')
state = ck.get('model', ck.get('state_dict', ck)) if isinstance(ck, dict) else ck
miss = model.load_state_dict(state, strict=False)
print(f"[load] missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)}")
model.eval()


@torch.no_grad()
def extract(samples):
    loader = build_eval_loader(samples, args)
    feats, pids, cams = [], [], []
    for batch in loader:
        gf = model(batch['img'].to(device, non_blocking=True))
        if isinstance(gf, (tuple, list)):
            gf = gf[0]
        feats.append(gf.detach().cpu())
        pids.append(np.asarray(batch['pid']))
        cams.append(np.asarray(batch['camid']))
    return (torch.cat(feats, 0), np.concatenate(pids), np.concatenate(cams))


def altitude_of(s):
    folder = os.path.basename(os.path.dirname(s['img_path']))
    m = re.search(r'A(\d)', folder)
    return int(m.group(1)) if m else -1


_scache = {}
def area_of(s):
    p = s['img_path']
    if p not in _scache:
        try:
            w, h = Image.open(p).size
            _scache[p] = float(h * w)
        except Exception:
            _scache[p] = -1.0
    return _scache[p]


def per_query_ap_rank(sim, qp, gp):
    """Return per-query AP and the rank(0-based) of the first true match."""
    order = np.argsort(-sim, axis=1)
    aps, first_rank, top1_wrong = [], [], []
    for i in range(sim.shape[0]):
        g = gp[order[i]]
        match = (g == qp[i]).astype(np.int32)
        if match.sum() == 0:
            aps.append(0.0); first_rank.append(-1); top1_wrong.append(order[i][0]); continue
        cum = np.cumsum(match)
        prec = cum / (np.arange(len(match)) + 1.0)
        ap = (prec * match).sum() / match.sum()
        aps.append(float(ap))
        fr = int(np.argmax(match))          # first index where match==1
        first_rank.append(fr)
        top1_wrong.append(order[i][0] if match[0] == 0 else -1)
    return np.array(aps), np.array(first_rank), np.array(top1_wrong), order


for tag, (qv, gv) in {'A->G': ('Aerial', 'Ground'), 'G->A': ('Ground', 'Aerial')}.items():
    q = filter_by_view(ds.query, qv)
    g = filter_by_view(ds.gallery, gv)
    qf, qp, qc = extract(q)
    gf, gp, gc = extract(g)
    qf = F.normalize(qf, dim=1); gf = F.normalize(gf, dim=1)
    sim = (qf @ gf.t()).numpy()
    aps, first_rank, top1_wrong, order = per_query_ap_rank(sim, qp, gp)

    q_alt = np.array([altitude_of(s) for s in q])
    q_area = np.array([area_of(s) for s in q])
    g_area = np.array([area_of(s) for s in g])

    print(f"\n========== {tag}  (Nq={len(q)} Ng={len(g)}) ==========")
    print(f"overall mAP = {100*aps.mean():.2f}")

    # (a) per query-altitude mAP
    print("(a) 按 query altitude 分桶 mAP:")
    for a in sorted(set(q_alt.tolist())):
        m = q_alt == a
        if m.sum() == 0: continue
        print(f"    altitude={a} (n={m.sum():4d}): mAP={100*aps[m].mean():.2f}")

    # (b) top-1 假阳 scale 比(几何相似负样本?)
    fp = top1_wrong >= 0
    if fp.sum() > 0:
        ratio = q_area[fp] / np.maximum(g_area[top1_wrong[fp]], 1.0)
        ratio = np.where(ratio < 1, 1.0 / ratio, ratio)   # symmetric >=1, 1=同尺度
        print(f"(b) top-1 假阳 n={fp.sum()}: scale比中位={np.median(ratio):.2f} "
              f"(<1.5 占 {100*np.mean(ratio<1.5):.0f}%  = 几何相似负样本)")

    # (c) 真值 first-rank vs query-真值 scale 差
    has = first_rank >= 0
    tm_idx = np.array([order[i][first_rank[i]] if first_rank[i] >= 0 else -1
                       for i in range(len(q))])
    tm_ratio = np.where(has, q_area / np.maximum(g_area[np.maximum(tm_idx, 0)], 1.0), np.nan)
    tm_ratio = np.where(tm_ratio < 1, 1.0/np.maximum(tm_ratio, 1e-6), tm_ratio)
    hard = has & (first_rank > 0)            # 真值不在 top-1 = 被推远
    print(f"(c) 真值被推离 top-1 的 n={hard.sum()}/{has.sum()}: "
          f"这些 hard-positive 的 query/真值 scale 比中位={np.nanmedian(tm_ratio[hard]):.2f} "
          f"vs top-1 命中的={np.nanmedian(tm_ratio[has&(first_rank==0)]):.2f}")

    # (d) 几何歧义子集 vs 全集
    alt_hi = q_alt >= 2
    big_scale = has & (tm_ratio >= np.nanmedian(tm_ratio[has]))
    amb = alt_hi | big_scale
    if amb.sum() > 0 and (~amb).sum() > 0:
        print(f"(d) 几何歧义子集(高altitude OR 大scale差) n={amb.sum()}: "
              f"mAP={100*aps[amb].mean():.2f}  vs 其余={100*aps[~amb].mean():.2f}  "
              f"gap={100*(aps[~amb].mean()-aps[amb].mean()):.2f}")

print("\n[done] ① kill-switch: 若(a)高度↑难↑明显 + (b)假阳多几何相似 + (d)歧义子集 mAP 显著低 → ① 成立")
