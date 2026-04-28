# exp319_tiny_oasdW05_s42 — Tiny OD Full + POSE_OA_SD_WEIGHT 0.5

- 机器: srvC (5060Ti)
- Config: `prcv_best_tiny.yml` + CLI `SOLVER.SEED 42 MODEL.POSE_OA_SD_WEIGHT 0.5 TEST.IMS_PER_BATCH 128`
- 启动: 2026-04-28 12:11 server, FINAL: 23:07 (~11h)
- 动机: OA-SD self-distillation 减半 (与 exp316 oasdW=2.0 互为反向 sweep)

## FINAL (e120)

- **eq+flip**: mAP **65.8%**, R1 **76.8%**
- **Global cosine+flip**: 65.7 / 76.4
- **MaxSim hybrid+flip**: **67.1 / 78.1**

## 对照 vs exp261 baseline (default oasdW=1.0)

| 指标 | exp319 | exp261 | Δ |
|------|--------|--------|----|
| eq+flip | 65.8/76.8 | 65.9/77.4 | -0.1/-0.6 |
| Global+flip | 65.7/76.4 | 65.8/76.0 | -0.1/+0.4 |
| **MaxSim** | **67.1/78.1** | **67.2/78.6** | **-0.1/-0.5** |

## 结论

POSE_OA_SD_WEIGHT 0.5 (减半) MaxSim **-0.1/-0.5**。slight neg。

加上 exp316 (oasdW=2.0) MaxSim 0/-0.6, OA-SD weight default 1.0 验证为 sweet spot, 上下 1× 都微负。
