# exp299_base_lr8_noplboa_s41 monitor — Base OD PLBOA OFF ablation

- 机器: srvC (5060Ti 16G)
- Config: `prcv_best_base.yml` + `SOLVER.SEED 41 SOLVER.BASE_LR 0.0008 MODEL.POSE_LOWER_BODY_OCC False TEST.IMS_PER_BATCH 64`
- 启动: 2026-04-24 23:42 CST, FINAL: 2026-04-25 22:12:03 CST (~22.5h)
- 动机: 测试 OD 上 PLBOA (Probabilistic Lower-Body Occlusion Augmentation) 关掉 vs 开启的对比, 类比 Market 上 PLBOA OFF 显著优于 ON 的结论

## FINAL (e120)

- **eq+flip**: mAP **70.9%**, R1 **78.0%**, R5 88.1%, R10 91.1%
- **Global cosine+flip**: 69.2 / 77.9
- **MaxSim hybrid+flip**: **72.7 / 80.5**

## 🎯 对照 exp296 LR8 PLBOA ON (lab4090, baseline)

| Metric | exp296 PLBOA ON | **exp299 PLBOA OFF** | Δ (OFF − ON) |
|--------|-----------------|----------------------|---------------|
| eq+flip mAP | 73.7 | **70.9** | **-2.8** |
| eq+flip R1 | 81.7 | 78.0 | **-3.7** |
| MaxSim+flip mAP | 74.9 | **72.7** | **-2.2** |
| MaxSim+flip R1 | 83.8 | 80.5 | **-3.3** |

**结论 (强 claim)**: **OD 上 PLBOA 是 net positive +2.8 mAP / +3.7 R1** (eq+flip) 或 **+2.2 mAP / +3.3 R1** (MaxSim+flip)。

## 跨数据集 PLBOA 效应对比 (paper claim)

| 数据集 | PLBOA OFF (in-domain) | PLBOA ON (in-domain) | Δ ON−OFF (in-domain mAP) | PLBOA ON 对跨域 (Occ-ReID via Market) |
|---------|----------------------|----------------------|---------------------------|-----------------------------------------|
| **Occ-Duke** | exp299: 72.7 (MaxSim) | exp296: 74.9 | **+2.2 mAP** ✓ helpful | (N/A, Market source) |
| **Market** | exp269b: 94.6 (MaxSim) | exp293: 93.9 | **-0.7 mAP** ✗ harmful | exp269: 88.2 → exp293: 62.8 (**-25.4** disaster) |

**Paper claim**: **PLBOA dataset-specific** — OD-train 应启用 (高 occlusion 训练数据下 PLBOA augment 学到更好 occlusion-aware features), Market-train 应关闭 (低 occlusion 训练数据 + PLBOA 强行注入 lower-body occlusion → overfit synthetic occluder pattern → 跨域 catastrophic 25 mAP 损失)。

## 训练轨迹

| Epoch | mAP |
|-------|-----|
| 10 | 53.5 |
| 20 | 61.9 |
| 30 | 66.0 |
| 40 | 68.2 |
| 50 | 69.5 |
| 60 | 70.1 |
| 70 | 69.3 (dip) |
| 80 | 71.1 |
| 90 | 70.9 |
| 100 | 70.8 |
| 110 | 70.8 |
| **120 FINAL** | **70.9** |

vs exp296 同期 e60-e120 一直 plateau 71-72 (约 -2.5 mAP 全程落后)。
