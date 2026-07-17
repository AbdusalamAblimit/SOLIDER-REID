# exp313_tiny_partW2_s42 — Tiny OD Full + POSE_PART_WEIGHT 2.0

- 机器: srvA (5060Ti)
- Config: `prcv_best_tiny.yml` + CLI `SOLVER.SEED 42 MODEL.POSE_PART_WEIGHT 2.0 TEST.IMS_PER_BATCH 128`
- 启动: 2026-04-28 01:13 my time, FINAL: ~12:00 server time (srvA 11:54)
- 动机: ID-side favor part (w_p=2/3 w_g=1/3), 测 ID balance 偏 part 是否帮助

## FINAL (e120)

- **eq+flip**: mAP **65.8%**, R1 **77.0%**, R5 86.7%, R10 89.5%
- **Global cosine+flip**: 65.2 / 75.4
- **MaxSim hybrid+flip**: **66.9 / 77.9** (srvA 恢复后补 eval, 2026-04-28)

## 训练轨迹 (eq+flip)

| Epoch | mAP | R1 |
|-------|-----|-----|
| 10 | 43.2 | 57.4 |
| 20 | 52.7 | 65.5 |
| 30 | 57.7 | 69.2 |
| 40 | 61.2 | 73.2 |
| 50 | 63.4 | 74.9 |
| 60 | 63.7 | 75.2 |
| 70 | 64.6 | 75.8 |
| 80 | 64.7 | 75.7 |
| 90 | 65.7 | 77.5 |
| 100 | 65.8 | 77.2 |
| 110 | 65.8 | 77.1 |
| **120 FINAL** | **65.8** | **77.0** |

## 对照

vs exp261 baseline (POSE_PART_WEIGHT 1.0, default): 65.9/77.4 eq, 67.2/78.6 MaxSim
- eq Δ: -0.1/-0.4 (噪声内)
- MaxSim Δ: **-0.3/-0.7** slight 负面

## 结论

POSE_PART_WEIGHT 2.0 (favor part, w_p=2/3 w_g=1/3) 在 Tiny 上:
- eq+flip slight neutral (-0.1/-0.4)
- **MaxSim 微 negative (-0.3/-0.7)**

加上 exp314 partW=0.5 (favor global) MaxSim 持平 baseline (0/0), 论 POSE_PART_WEIGHT 1.0 (default symmetric) 是 sweet spot, 偏 part 微负, 偏 global 中性。
