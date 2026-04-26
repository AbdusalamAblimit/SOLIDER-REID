# exp305_full_noLGPA_t_od_s42 monitor — Phase 3-D Tiny OD Full − LGPA s42

- 机器: lab4090 (24GB 4090, mmpose-abu env)
- Config: `prcv_best_tiny.yml` + CLI `SOLVER.SEED 42 MODEL.POSE_LGPA False TEST.IMS_PER_BATCH 128`
- 启动: 2026-04-26 09:42 CST, FINAL: 14:00 CST (~4.3h)
- Scaffold: Swin-Tiny + GCN512 + OA-SD + ParAug + PLBOA + 2-stage PSG (**LGPA OFF**)
- 动机: **Phase 3-D Tiny mirror of exp301 (Small no-LGPA)**, 验证 LGPA ablation 在 3 个 backbone 一致性

## 训练轨迹

| Epoch | mAP | R1 |
|-------|-----|----|
| 10 | 44.0 | — |
| 20 | 53.9 | — |
| 30 | 58.8 | — |
| 40 | 60.7 | — |
| 50 | 62.4 | — |
| 60 | 60.6 (dip) | — |
| 70 | 63.3 | — |
| 80 | 64.3 | — |
| 90 | 64.5 | — |
| 100 | 64.4 | — |
| 110 | (need check) | — |
| **120 FINAL** | **64.5** | **76.0** |

## FINAL (2026-04-26 14:00 CST)

- **eq+flip**: mAP **64.5%**, R1 **76.0%**, R5 86.2%, R10 89.2%
- **Global cosine+flip**: 65.7 / 76.2
- **MaxSim hybrid+flip**: **64.5 / 76.0** (vs exp261 67.2/78.6 → **-2.7 / -2.6**)
- 注: MaxSim = eq+flip, **MaxSim 无 boost**, 一致于 exp301 pattern (LGPA 是 MaxSim 主驱动)

## 🎯 Phase 3-D 3-backbone 完整 (LGPA ablation)

| Backbone | Exp (no-LGPA) | eq+flip | vs Full Scaffold (eq+flip) | Δ mAP |
|----------|---------------|---------|----------------------------|--------|
| **Tiny** | exp305 (本) | **64.5 / 76.0** | exp261: 65.9/77.4 | **-1.4 / -1.4** |
| **Small** | exp301 | 71.9/83.0 | exp285b: 73.8/83.8 | **-1.9 / -0.8** |
| **Base** | (pending) | — | exp263d: 74.1/83.3 | — |

**结论**:
- LGPA 在 Tiny/Small 上贡献 **+1.4 ~ +1.9 mAP** (eq+flip)
- 中等正贡献, 比 GCN 冗余 (0 mAP) 显著
- Paper Phase 3-D 完整 3-backbone 后写: "LGPA contributes consistent +1.5-2.0 mAP across all backbones; in contrast, GCN contributes 0 mAP — LGPA is the dominant pose-conditioned semantic component"

## 训练观察

- e60 dip 60.6 (vs e50 62.4), 单点 noise, e70 立即恢复 63.3
- e80-e120 plateau 在 64.3-64.5
- 整体 trajectory 健康
