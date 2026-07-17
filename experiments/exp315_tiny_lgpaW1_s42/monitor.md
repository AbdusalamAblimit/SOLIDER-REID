# exp315_tiny_lgpaW1_s42 — Tiny OD Full + POSE_LGPA_ASSIGN_WEIGHT 1.0

- 机器: srvC (5060Ti)
- Config: `prcv_best_tiny.yml` + CLI `SOLVER.SEED 42 MODEL.POSE_LGPA_ASSIGN_WEIGHT 1.0 TEST.IMS_PER_BATCH 128`
- 启动: 2026-04-28 01:14 my time, FINAL: 12:06 server time
- 动机: LGPA aux loss weight 翻倍 (0.5 → 1.0), 测 LGPA 监督加强是否帮助 (Phase 3-D 已证 LGPA 关键)

## FINAL (e120)

- **eq+flip**: mAP **65.8%**, R1 **76.9%**, R5 86.6%, R10 89.6%
- **Global cosine+flip**: 65.7 / 75.9
- **MaxSim hybrid+flip**: **67.0 / 77.4**

## 训练轨迹 (eq+flip)

| Epoch | mAP | R1 |
|-------|-----|-----|
| 10 | 43.4 | 58.1 |
| 20 | 53.5 | 67.1 |
| 30 | 58.6 | 71.5 |
| 40 | 62.0 | 74.5 |
| 50 | 63.6 | 75.3 |
| 60 | 64.0 | 75.9 |
| 70 | 64.6 | 76.1 |
| 80 | 64.8 | 76.5 |
| 90 | 65.7 | 77.3 |
| 100 | 65.6 | 77.0 |
| 110 | 65.7 | 77.1 |
| **120 FINAL** | **65.8** | **76.9** |

## 对照

vs exp261 baseline (default 0.5): 65.9/77.4 eq, **67.2/78.6 MaxSim**

| 指标 | exp315 | exp261 | Δ |
|------|--------|--------|----|
| eq+flip | 65.8/76.9 | 65.9/77.4 | -0.1/-0.5 |
| Global+flip | 65.7/75.9 | 65.8/76.0 | -0.1/-0.1 |
| **MaxSim** | **67.0/77.4** | **67.2/78.6** | **-0.2/-1.2** |

## 结论

POSE_LGPA_ASSIGN_WEIGHT 1.0 (vs default 0.5) MaxSim **net -0.2 mAP / -1.2 R1**。slight net negative。LGPA aux 加倍 hurt R1 显著, mAP 在 noise 内。

Default 0.5 是 sweet spot, 不需要加大 LGPA aux 监督。
