# exp314_tiny_partW05_s42 — Tiny OD Full + POSE_PART_WEIGHT 0.5

- 机器: srvB (5060Ti)
- Config: `prcv_best_tiny.yml` + CLI `SOLVER.SEED 42 MODEL.POSE_PART_WEIGHT 0.5 TEST.IMS_PER_BATCH 128`
- 启动: 2026-04-28 01:13 my time, FINAL: 11:58 server time
- 动机: ID-side favor global (w_g=2/3 w_p=1/3), 测 ID balance 偏 global 是否帮助

## FINAL (e120)

- **eq+flip**: mAP **65.8%**, R1 **77.5%**, R5 87.1%, R10 89.7%
- **Global cosine+flip**: 66.0 / 76.5
- **MaxSim hybrid+flip**: **67.2 / 78.6** ← 与 baseline 完全相等!

## 训练轨迹 (eq+flip)

| Epoch | mAP | R1 |
|-------|-----|-----|
| 10 | 43.4 | 57.3 |
| 20 | 55.4 | 68.8 |
| 30 | 60.2 | 73.1 |
| 40 | 62.4 | 74.8 |
| 50 | 63.6 | 75.1 |
| 60 | 64.2 | 77.1 |
| 70 | 64.6 | 77.1 |
| 80 | 65.0 | 77.2 |
| 90 | 65.8 | 77.4 |
| 100 | 65.6 | 77.5 |
| 110 | 65.8 | 77.4 |
| **120 FINAL** | **65.8** | **77.5** |

## 对照

vs exp261 baseline (default 1.0): 65.9/77.4 eq, **67.2/78.6 MaxSim**

| 指标 | exp314 | exp261 | Δ |
|------|--------|--------|----|
| eq+flip | 65.8/77.5 | 65.9/77.4 | -0.1/+0.1 |
| Global+flip | 66.0/76.5 | 65.8/76.0 | +0.2/+0.5 |
| **MaxSim** | **67.2/78.6** | **67.2/78.6** | **0/0** |

## 结论

POSE_PART_WEIGHT 0.5 (favor global) MaxSim **完全等于** baseline 67.2/78.6。R1 (eq+flip) +0.1, mAP -0.1。

总体: **net neutral**, 可能略微 R1 受益但 mAP 略损, 没有显著差异。Default (1.0) 已经合理。
