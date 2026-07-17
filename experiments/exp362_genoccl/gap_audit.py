#!/usr/bin/env python3
"""exp362 gap-measured occlusion engine — Step 1: gap audit (cheap, no training, no diffusion).

codex pivot 窄缝是 "gap-measured occlusion distribution engine"。第一步先验证前提：
occluded_duke train 的遮挡分布 vs query(test) 的遮挡分布到底有没有 gap。
- 有 gap（query 某部位遮挡 >> train）→ 生成引擎有的放矢（合成 query-like 遮挡补 train 分布）。
- gap 小 → 生成引擎前提弱，train 已经覆盖 test 遮挡分布，不值得做。

用 pose visibility 测 per-body-group 遮挡频率（整组 kp visibility < thr = 该部位被遮）。
也按 #可见组 数测"遮挡严重度"分布（heavy-occ：可见组 <=2）。
"""
import numpy as np, os, sys
REPO = sys.argv[1] if len(sys.argv) > 1 else '/home/afr/SOLIDER-REID'
DATA = os.path.join(REPO, 'data/occluded_duke')
GROUPS = {'head': [0, 1, 2, 3, 4], 'torso': [5, 6, 11, 12], 'larm': [5, 7, 9], 'rarm': [6, 8, 10],
          'legs': [11, 12, 13, 14, 15, 16]}
GK = list(GROUPS)
VIS_THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3   # threshold sweep (codex: 0.3/0.5/0.7)


def occ_profile(npz):
    d = np.load(os.path.join(DATA, npz), allow_pickle=True)
    vis = d['visibility'].astype(np.float32)                     # [N,17] 0-1 score
    N = len(vis)
    gocc = np.zeros((N, len(GK)), bool)
    for gi, gk in enumerate(GK):
        gocc[:, gi] = (vis[:, GROUPS[gk]] < VIS_THR).all(1)     # 整组被遮
    group_freq = gocc.mean(0)                                    # per-group 遮挡频率
    nvis_groups = (~gocc).sum(1)                                 # 每图可见组数 [N]
    heavy = (nvis_groups <= 2).mean()                            # heavy-occ 比例(可见组<=2)
    return group_freq, heavy, N, nvis_groups


def main():
    for q in ['pose_train.npz', 'pose_query.npz', 'pose_gallery.npz']:
        if not os.path.exists(os.path.join(DATA, q)):
            print(f"[ERR] {q} missing"); return
    tf, th, ntr, tnv = occ_profile('pose_train.npz')
    qf, qh, nq, qnv = occ_profile('pose_query.npz')
    print(f"[gap audit] train N={ntr}  query N={nq}  vis_thr={VIS_THR}\n")
    print(f"{'group':8}{'train_occ%':>11}{'query_occ%':>11}{'gap(q-t)':>11}")
    for gi, gk in enumerate(GK):
        print(f"{gk:8}{tf[gi]*100:11.1f}{qf[gi]*100:11.1f}{(qf[gi]-tf[gi])*100:+11.1f}")
    print(f"\nheavy-occ(可见组<=2)比例:  train={th*100:.1f}%  query={qh*100:.1f}%  gap={ (qh-th)*100:+.1f}%")
    print(f"可见组数分布 train: {np.bincount(tnv, minlength=6)[:6]}")
    print(f"可见组数分布 query: {np.bincount(qnv, minlength=6)[:6]}")
    gaps = qf - tf
    gi = int(np.argmax(np.abs(gaps)))
    maxg = gaps[gi] * 100
    verdict = 'GAP EXISTS — 生成引擎有的放矢(合成 query-like 遮挡补 train)' if (abs(maxg) > 5 or abs(qh - th) > 0.05) \
        else 'gap 小 — train 已覆盖 test 遮挡分布, 生成引擎前提弱'
    print(f"\n[verdict] 最大 per-group gap={maxg:+.1f}% ({GK[gi]}); heavy-occ gap={(qh-th)*100:+.1f}% → {verdict}")
    print("[done] gap audit complete.")


if __name__ == '__main__':
    main()
