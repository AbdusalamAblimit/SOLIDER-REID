# exp303_full_t_od_lr4_s41 monitor — Tiny OD LR sweep LR4

- 机器: srvB (5060Ti)
- Config: `prcv_best_tiny.yml` + CLI `SOLVER.SEED 41 SOLVER.BASE_LR 0.0004 TEST.IMS_PER_BATCH 128`
- 启动: 2026-04-26 03:30 CST, FINAL: 14:20:03 CST (~10.8h)
- 动机: Tiny LR sweep — exp261 Tiny LR8 default 65.9/77.4, 测 LR4 是否 underfit 或持平

## FINAL (e120)

- **eq+flip**: mAP **64.4%**, R1 **74.8%**, R5 ?, R10 ?
- **Global cosine+flip**: 64.0 / 73.3
- **MaxSim hybrid+flip**: **65.7 / 76.1**

## 对照 Tiny LR sweep

| Exp | LR | eq+flip | MaxSim+flip |
|-----|-----|---------|-------------|
| exp261 | 8e-4 | 65.9/77.4 | 67.2/78.6 |
| **exp303 (本)** | **4e-4** | **64.4/74.8** | **65.7/76.1** |
| Δ vs exp261 | | **-1.5/-2.6** | **-1.5/-2.5** |

**结论**: Tiny LR4 比 LR8 underfit -1.5 mAP, R1 -2.5. LR8 仍 sweet spot for Tiny (与 Base LR sweep 一致结论)。

## 训练轨迹 (mAP)

| Epoch | mAP | R1 |
|-------|-----|----|
| 10 | 40.1 | — |
| 20 | 49.1 | — |
| 30 | 56.7 | — |
| 40 | 59.7 | — |
| 50 | 60.9 | — |
| 60 | 62.1 | — |
| 70 | 63.1 | — |
| 80 | 63.8 | 74.8 |
| 90 | 64.1 | 75.2 |
| 100 | 64.4 | 75.1 |
| 110 | 64.4 | 74.9 |
| **120 FINAL** | **64.4** | **74.8** |

e80 后 plateau (64.4 持续), no further gain。
