# exp312_tiny_globalScale2_s42 — Tiny OD Full + GLOBAL_LOSS_SCALE 2.0

- 机器: lab4090
- Config: `prcv_best_tiny.yml` + CLI `SOLVER.SEED 42 MODEL.GLOBAL_LOSS_SCALE 2.0 TEST.IMS_PER_BATCH 64`
- 启动: 2026-04-27 17:13 server time, FINAL: 21:08 (~4h)
- 动机: 测试 GLOBAL_LOSS_SCALE 反方向 (0.5 已证负, 试 2.0)

## FINAL (e120)

| 指标 | exp312 (GLOBAL 2.0) | exp261 baseline (1.0) | Δ |
|------|---------------------|------------------------|----|
| eq+flip | **65.7/76.6** | 65.9/77.4 | -0.2/-0.8 |
| Global cosine+flip | 65.4/75.3 | 65.8/76.0 | -0.4/-0.7 |
| **MaxSim hybrid+flip** | **66.8/77.2** | **67.2/78.6** | **-0.4/-1.4** |

## 训练轨迹 (eq+flip)

| Epoch | mAP | R1 |
|-------|-----|-----|
| 10 | 45.7 | 59.3 |
| 20 | 54.0 | 67.9 |
| 30 | 60.3 | 72.4 |
| 40 | 61.9 | 74.6 |
| 50 | 62.8 | 74.6 |
| 60 | 63.5 | 75.0 |
| 70 | 63.9 | 75.2 |
| 80 | 64.6 | 75.8 |
| 90 | 65.3 | 76.7 |
| 100 | 65.4 | 76.3 |
| 110 | 65.6 | 76.5 |
| **120 FINAL** | **65.7** | **76.6** |

## 结论

GLOBAL_LOSS_SCALE 2.0 在 Tiny 上 **-0.4 mAP / -1.4 R1 MaxSim** vs default (effective 1.0)。

结合 exp311b (0.5 真生效, Small s1234) **-0.7 mAP MaxSim**, 两个方向都 net negative:

| Scale | mAP MaxSim Δ | 结论 |
|-------|---------------|------|
| 0.5 (exp311b) | -0.7 (Small) | net negative |
| **1.0 (default)** | **0 (baseline)** | **sweet spot** ⭐ |
| 2.0 (exp312) | -0.4 (Tiny) | net negative |

GLOBAL_LOSS_SCALE 应保持 1.0, **不需要调** 在论文中。
