# exp298_base_lr2_s41 monitor — Base OD LR sweep LR2 (floor)

- 机器: srvB (5060Ti 16G)
- Config: `prcv_best_base.yml` + `SOLVER.SEED 41 SOLVER.BASE_LR 0.0002 TEST.IMS_PER_BATCH 64`
- 启动: 2026-04-24 23:42 CST, FINAL: 2026-04-25 22:04:03 CST (~22h)
- 动机: LR sweep floor — LR2 是否严重 underfit

## FINAL (e120)

- **eq+flip**: mAP **68.6%**, R1 **78.6%**, R5 87.5%, R10 90.2%
- **Global cosine+flip**: 67.5 / 75.0
- **MaxSim hybrid+flip**: **69.6 / 79.1**

## 对照 (vs LR8 baseline exp296)

| Metric | exp296 LR8 | **exp298 LR2** | Δ |
|--------|------------|-----------------|----|
| eq+flip mAP | 73.7 | **68.6** | **-5.1** |
| eq+flip R1 | 81.7 | **78.6** | -3.1 |
| MaxSim+flip mAP | 74.9 | **69.6** | **-5.3** |
| MaxSim+flip R1 | 83.8 | **79.1** | -4.7 |

**结论**: LR2 严重 underfit, mAP 损 5.1-5.3, **paper 写为 LR ablation 下界**, 证明 LR8 sweet spot 不能再降。

## 训练轨迹

| Epoch | mAP | 备注 |
|-------|-----|------|
| 10 | 1.3 | 初期 warmup 起步极慢, 几乎随机 |
| 20 | 44.6 | warmup 后期开始学习 |
| 30 | 57.8 | |
| 40 | 63.3 | |
| 50 | 66.0 | |
| 60 | 66.4 | |
| 70 | 67.5 | |
| 80 | 68.3 | |
| 90 | 68.4 | |
| 100 | 68.5 | |
| 110 | 68.6 | |
| **120 FINAL** | **68.6** | plateau 严重 underfit |

e10 mAP 1.3% near-random 是 LR2 + WARMUP_EPOCHS 20 cosine 调度下学习率 e10 太小. 后期 LR 上去后开始学习, 但总体不及 LR8。
