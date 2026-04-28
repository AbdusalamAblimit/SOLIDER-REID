# exp318_tiny_partTriW05_s42 — Tiny OD Full + POSE_PART_TRI_WEIGHT 0.5

- 机器: srvB (5060Ti)
- Config: `prcv_best_tiny.yml` + CLI `SOLVER.SEED 42 MODEL.POSE_PART_TRI_WEIGHT 0.5 TEST.IMS_PER_BATCH 128`
- 启动: 2026-04-28 12:11 server, FINAL: 23:08 (~11h)
- 动机: Triplet-side favor global (wt_g=2/3 wt_p=1/3, 与 ID-side exp314 同方向但 Tri 端)

## FINAL (e120)

- **eq+flip**: mAP **65.9%**, R1 **77.7%**
- **Global cosine+flip**: 65.8 / 76.3
- **MaxSim hybrid+flip**: **67.1 / 78.3**

## 对照 vs exp261 baseline (default partTriW=1.0)

| 指标 | exp318 | exp261 | Δ |
|------|--------|--------|----|
| eq+flip | 65.9/77.7 | 65.9/77.4 | 0/+0.3 |
| Global+flip | 65.8/76.3 | 65.8/76.0 | 0/+0.3 |
| **MaxSim** | **67.1/78.3** | **67.2/78.6** | **-0.1/-0.3** |

## 结论

POSE_PART_TRI_WEIGHT 0.5 (Triplet-side favor global) MaxSim **slight neg -0.1/-0.3**, eq+flip R1 +0.3。Net neutral / slightly negative。

加上 exp314 partW=0.5 (ID-side, MaxSim 67.2/78.6 = baseline), 论 default 是双 sweet spot, 偏 global 任一边都不显著改进。
