# exp316_tiny_oasdW2_s42 — Tiny OD Full + POSE_OA_SD_WEIGHT 2.0

- 机器: lab4090
- Config: `prcv_best_tiny.yml` + CLI `SOLVER.SEED 42 MODEL.POSE_OA_SD_WEIGHT 2.0 TEST.IMS_PER_BATCH 64`
- 启动: 2026-04-28 02:30 server, FINAL: 06:22 (~3h52m)
- 动机: OA-SD self-distillation weight 翻倍 (1.0 → 2.0), 测加强自蒸馏是否帮助

## FINAL (e120)

- **eq+flip**: mAP **66.0%**, R1 **77.6%**, R5 87.1%, R10 89.8%
- **Global cosine+flip**: 65.7 / 75.7
- **MaxSim hybrid+flip**: **67.2 / 78.0**

## 训练轨迹 (eq+flip)

| Epoch | mAP | R1 |
|-------|-----|-----|
| 10 | 43.6 | 57.1 |
| 20 | 54.5 | 68.0 |
| 30 | 59.5 | 72.2 |
| 40 | 62.7 | 74.8 |
| 50 | 63.9 | 76.3 |
| 60 | 64.2 | 76.3 |
| 70 | 64.5 | 76.2 |
| 80 | 65.1 | 77.5 |
| 90 | 66.0 | 78.1 |
| 100 | 65.7 | 77.6 |
| 110 | 65.9 | 77.6 |
| **120 FINAL** | **66.0** | **77.6** |

## 对照 vs exp261 baseline (default oasdW=1.0)

| 指标 | exp316 (oasdW=2.0) | exp261 baseline | Δ |
|------|--------------------|------------------|----|
| eq+flip | 66.0/77.6 | 65.9/77.4 | +0.1/+0.2 |
| Global+flip | 65.7/75.7 | 65.8/76.0 | -0.1/-0.3 |
| **MaxSim** | **67.2/78.0** | **67.2/78.6** | **0/-0.6** |

## 结论

OA-SD weight 2.0 在 Tiny 上 **net neutral**: eq slight +, MaxSim mAP =, MaxSim R1 -0.6。Default 1.0 is sweet spot for Tiny。

⚠️ 注意 e90 单点 66.0/78.1 是峰值, e120 FINAL 仅持平 baseline。可能是 trajectory noise, 也可能是 cosine 末段过度衰减。
