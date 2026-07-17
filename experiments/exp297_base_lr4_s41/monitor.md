# exp297_base_lr4_s41 monitor — Base OD LR sweep LR4

- 机器: srvA (5060Ti 16G)
- Config: `prcv_best_base.yml` + `SOLVER.SEED 41 SOLVER.BASE_LR 0.0004 TEST.IMS_PER_BATCH 64`
- 启动: 2026-04-24 23:42 CST, FINAL: 2026-04-25 21:45:43 CST (~22h)
- Speed: 5060Ti ~11min/epoch
- 动机: LR sweep LR4 — 测试历史 exp260 LR4 (72.6) underfit conclusion 在 v2 code 下是否仍成立

## FINAL (e120)

- **eq+flip**: mAP **73.2%**, R1 **82.4%**, R5 90.2%, R10 92.6%
- **Global cosine+flip**: 73.3 / 82.2
- **MaxSim hybrid+flip**: **74.6 / 84.1**

## 对照

| Exp | LR | eq+flip | MaxSim+flip | Δ vs exp296 LR8 |
|-----|----|--------|-------------|------------------|
| exp296 (lab4090) | 8e-4 | 73.7/81.7 | 74.9/83.8 | baseline |
| **exp297 (srvA)** | **4e-4** | **73.2/82.4** | **74.6/84.1** | -0.5 / +0.7 (eq), -0.3 / +0.3 (MaxSim) |
| exp298 (srvB) | 2e-4 | 68.6/78.6 | 69.6/79.1 | -5.1 / -3.1 |

**结论**: LR4 vs LR8 同环境 (5060Ti vs lab4090 还有 GPU 差异), MaxSim mAP -0.3, R1 +0.3, **接近 tie**, **不是显著 underfit**。比 exp260 historical LR4 (72.6) 高 0.6 mAP, code 改进让 LR4 表现接近 LR8。

## 训练轨迹

| Epoch | mAP | R1 |
|-------|-----|----|
| 10 | 52.3 | 61.6 |
| 20 | 62.5 | 73.4 |
| 30 | 67.0 | — |
| 40 | 69.8 | — |
| 50 | 71.7 | — |
| 60 | 71.6 | — |
| 70 | 72.6 | — |
| 80 | 73.3 | — |
| 90 | 73.1 | — |
| 100 | 73.2 | — |
| 110 | 73.2 | — |
| **120 FINAL** | **73.2** | **82.4** |

整体 plateau e80 后稳定 73.1-73.3。
