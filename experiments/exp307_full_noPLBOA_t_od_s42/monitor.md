# exp307_full_noPLBOA_t_od_s42 monitor — Tiny OD Full **no PLBOA** s42

- 机器: srvB (5060Ti)
- Config: `prcv_best_tiny.yml` + CLI `SOLVER.SEED 42 MODEL.POSE_LOWER_BODY_OCC False TEST.IMS_PER_BATCH 128`
- 启动: 2026-04-26 ~12:00 CST, FINAL: 2026-04-27 02:51 CST (~14.9h)
- Scaffold: Swin-Tiny + GCN512 + LGPA-D + OA-SD + ParAug + 2-stage PSG (**PLBOA OFF**)
- 动机: **Tiny PLBOA 消融** — 配 exp299 Base no-PLBOA 对照, 验证 PLBOA 在不同 backbone 一致正贡献

## FINAL (e120, 2026-04-27 02:51 CST)

- **eq+flip (train log)**: mAP **62.8%**, R1 **71.8%**, R5 83.8%, R10 87.8%
- **Global cosine+flip**: 61.7 / 70.9
- **MaxSim hybrid+flip**: **64.5 / 73.5**

## 训练轨迹 (mAP, eq+flip)

| Epoch | mAP |
|-------|-----|
| 10 | 41.0 |
| 20 | 50.3 |
| 30 | 56.8 |
| 40 | 58.8 |
| 50 | 60.8 |
| 60 | 61.7 |
| 70 | 62.0 |
| 80 | 62.6 |
| 90 | 62.5 |
| 100 | 62.7 |
| 110 | 62.7 |
| **120 FINAL** | **62.8** |

e80 起 plateau 62.5-62.8 ± 0.3 噪声范围,基本无 further gain。

## 对照 — Tiny PLBOA ablation

| Backbone | PLBOA | Exp | eq+flip | Global+flip | **MaxSim+flip** |
|----------|-------|-----|---------|-------------|-----------------|
| Tiny | **ON** | exp261 | 65.9/77.4 | 65.8/76.0 | **67.2/78.6** |
| Tiny | **OFF** | **exp307 (本)** | **62.8/71.8** | 61.7/70.9 | **64.5/73.5** |
| Δ Tiny | | | **-3.1/-5.6** | -4.1/-5.1 | **-2.7/-5.1** |

## 跨 backbone 一致性 (PLBOA ablation on Occ-Duke)

| Backbone | Exp ON | Exp OFF | Δ MaxSim mAP | Δ MaxSim R1 |
|----------|--------|---------|---------------|---------------|
| **Tiny** | exp261 67.2/78.6 | **exp307** 64.5/73.5 | **+2.7** | **+5.1** |
| **Base** | exp296 74.9/83.8 | exp299 72.7/80.5 | **+2.2** | **+3.3** |

**核心 Paper claim — PLBOA dataset-specific (强化版)**:
- **Occ-Duke**: PLBOA ON > OFF by **+2.2-2.7 mAP MaxSim** across 2 backbones (Tiny + Base 一致 net positive)
- **Market**: PLBOA ON < OFF by -0.7 mAP (in-domain) AND -25.4 mAP (cross-domain Occ-ReID)
- 论文写: "PLBOA contributes consistent **+2.2-2.7 mAP** across backbones on Occ-Duke (occlusion-rich), but causes **-25 mAP catastrophic drop** on Market→Occ-ReID transfer; data-augmentation should be conditioned on the deployment domain occlusion rate"

## 结论

exp307 = **Tiny PLBOA 消融最后一格 FINAL**, 数字 62.8/71.8 eq, 64.5/73.5 MaxSim。PLBOA 在 Tiny 上贡献 +2.7 mAP MaxSim, 比 Base (+2.2 mAP) 略大 — Tiny 容量小, augmentation 帮助更显著。Paper Phase 3 PLBOA 段落 2-backbone evidence 完整。

## 下一步

srvB idle 后立即跑 **exp267 Tiny Market v2 重 eval** (验证 v1 numbers 是否需要更新)。
