# exp321b_small_lgpaW025_s1234 — Small OD Full + POSE_LGPA_ASSIGN_WEIGHT 0.25

- 机器: lab4090
- Config: `prcv_best_small.yml` + CLI `SOLVER.SEED 1234 MODEL.POSE_LGPA_ASSIGN_WEIGHT 0.25 TEST.IMS_PER_BATCH 64`
- 启动: 2026-04-28 12:34 server, FINAL: 18:40 (~6h)
- 动机: Tiny sweep winner exp317 (lgpaW=0.25, +0.2 mAP MaxSim) 在 Small 上验证, 测是否 transfer

## FINAL (e120)

- **eq+flip**: mAP **73.9%**, R1 **83.7%**
- **Global cosine+flip**: 73.4 / 83.3
- **MaxSim hybrid+flip**: **74.9 / 85.4**

## 训练轨迹 (eq+flip)

| Epoch | mAP | R1 |
|-------|-----|-----|
| 10 | 55.1 | 65.2 |
| 20 | 63.1 | 73.7 |
| 30 | 67.9 | 78.6 |
| 40 | 70.8 | 81.4 |
| 50 | 72.1 | 82.1 |
| 60 | 72.8 | 83.2 |
| 70 | 72.8 | 82.7 |
| 80 | 73.7 | 83.6 |
| 90 | 73.7 | 83.4 |
| 100 | 73.8 | 83.6 |
| 110 | 73.8 | 83.7 |
| **120 FINAL** | **73.9** | **83.7** |

## 对照 vs exp295 baseline (s1234, default lgpaW=0.5)

| 指标 | exp321b (lgpaW=0.25) | exp295 baseline | Δ |
|------|----------------------|-----------------|----|
| eq+flip | 73.9/83.7 | 74.2/84.0 | -0.3/-0.3 |
| Global+flip | 73.4/83.3 | 73.7/83.3 | -0.3/0 |
| **MaxSim** | **74.9/85.4** | **75.2/85.4** | **-0.3/0** |

## 结论 (重要)

**Tiny exp317 +0.2 mAP MaxSim 没在 Small 上重现** — exp321b Small s1234 lgpaW=0.25 MaxSim 比 baseline **slight -0.3 mAP**, R1 持平 (85.4 = 85.4)。

Tiny +0.2 大概率在 multi-seed std (0.42-0.45) 内, 不是真实 improvement。

**Paper 决定**: 不把 lgpaW=0.25 写为 paper improvement, 保持 default 0.5。

待 exp321c (s42) FINAL 后再确认。如 s42 也 = 或 < baseline, 彻底放弃这条路径。
