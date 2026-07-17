# exp320_small_lgpaNoDetach_s1234 — Small OD Full + POSE_LGPA_DETACH=False

- 机器: lab4090
- Config: `prcv_best_small.yml` + CLI `SOLVER.SEED 1234 MODEL.POSE_LGPA_DETACH False TEST.IMS_PER_BATCH 64`
- 启动: 2026-04-28 06:25 server, FINAL: 12:32 (~6h)
- 动机: SOTA push 探索 — 让 LGPA aux loss 反传到 backbone (default DETACH=True), 测是否可让 LGPA shape backbone features 提升 mAP

## FINAL (e120)

- **eq+flip**: mAP **68.1%**, R1 **79.3%**
- **Global cosine+flip**: 67.4 / 77.8
- **MaxSim hybrid+flip**: **68.8 / 79.6**

## 训练轨迹 (eq+flip)

| Epoch | mAP | R1 |
|-------|-----|-----|
| 10 | 46.0 | 59.1 |
| 20 | 55.1 | 65.7 |
| 30 | 62.7 | 73.7 |
| 40 | 66.4 | 77.1 |
| 50 | 67.2 | 77.6 |
| 60 | 67.8 | 78.6 |
| 70 | 67.6 | 79.6 |
| 80 | 68.3 | 79.4 |
| 90 | 67.9 | 79.2 |
| 100 | 68.1 | 79.7 |
| 110 | 68.1 | 79.5 |
| **120 FINAL** | **68.1** | **79.3** |

## 对照 vs exp295 baseline (s1234, default DETACH=True)

| 指标 | exp320 (DETACH=False) | exp295 baseline (True) | Δ |
|------|------------------------|------------------------|----|
| eq+flip | 68.1/79.3 | 74.2/84.0 | **-6.1/-4.7** |
| Global+flip | 67.4/77.8 | 73.7/83.3 | -6.3/-5.5 |
| **MaxSim** | **68.8/79.6** | **75.2/85.4** | **-6.4/-5.8** |

## 结论 (强 negative)

POSE_LGPA_DETACH=False 在 Small 上 **catastrophic -6.4 mAP MaxSim**。即让 LGPA 反传到 backbone 严重 hurt 学习 (e10 46.0% catastrophic underfit, e80 plateau 68.3 远低 baseline 74-75).

**Paper 可写**: "We find LGPA must be detached from backbone gradient flow. Allowing LGPA assignment loss to backprop into backbone causes severe underfitting (-6.4 mAP), confirming that LGPA serves as a downstream attention head over frozen pose-spatial-gated features rather than a backbone-shaping module"

POSE_LGPA_DETACH=True (current default) 是必要选择, 不是任意 hyperparam。
