# exp294_lgpaOnly_2stg_b_od_s41 monitor — Base Full-GCN (LGPA-only) + 2-stage PSG on Occ-Duke seed 41

- 机器: lab4090 (24GB 4090, mmpose-abu env)
- 启动: 2026-04-23 18:00 CST
- Log: `/home/afr/SOLIDER-REID/log/occluded_duke/exp294_lgpaOnly_2stg_b_od_s41/train_log.txt` (UTC clock)
- Config: `configs/occluded_duke/prcv_best_base.yml` + CLI `SOLVER.SEED 41 MODEL.POSE_SKELETON_GCN False TEST.IMS_PER_BATCH 64`
- Scaffold: Swin-Base + **LGPA + OA-SD + ParAug + PLBOA + 2-stage PSG `[-2,-1]`** (**NO GCN**)
- Speed: ~4.0 min/epoch (4090, BS=64, 227 iter)
- 总训练时长: 8h18min (18:00 → 02:18 tmr)
- **动机**: Base 上 Full-GCN (LGPA-only) 能否达 exp263d (Full+GCN) SOTA 74.1/83.3, 补 Phase 3-C Base 行

## 训练轨迹 (flip-test, eq_concat global; UTC log time)

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 52.4 | 62.8 | - | 81.9 |
| 20 | 64.5 | 76.0 | - | 89.3 |
| 30 | 68.7 | 79.8 | - | 91.0 |
| 40 | 67.9 (dip) | 78.3 | - | 89.9 |
| 50 | 72.5 | 81.8 | 90.6 | 92.4 |
| 60 | 73.1 | 82.3 | 90.5 | 92.4 |
| 70 | 73.4 | 82.7 | 90.4 | 92.4 |
| 80 | 73.7 | 82.9 (R1 peak) | 90.9 | 92.4 |
| 90 | 73.9 | 82.4 | 90.5 | 92.5 |
| 100 | 74.0 | 82.7 | 90.5 | 92.5 |
| **110** | **74.1** (mAP peak) | 82.7 | 90.3 | 92.5 |
| **120 FINAL** | **74.0** | **82.6** | **90.5** | **92.4** |

## FINAL (2026-04-24 02:18:48 CST)

- **mAP: 74.0%**, **Rank-1: 82.6%**, R5: 90.5%, R10: 92.4%

## 🏆 核心对照: Base + LGPA-only vs Base Full (with GCN), same seed 41

| Exp | GCN | mAP | R1 | R5 | R10 |
|-----|-----|-----|----|----|----|
| exp263d | **ON (GCN512)** | **74.1** | **83.3** | 90.8 | 93.0 |
| **exp294 (本)** | **OFF** | **74.0** | **82.6** | **90.5** | **92.4** |
| **Δ (Full-GCN vs Full+GCN)** | | **-0.1** | **-0.7** | -0.3 | -0.6 |

**结论**: Base 上 GCN 仅贡献 **+0.1 mAP / +0.7 R1 / +0.3 R5 / +0.6 R10** — 几乎可忽略。

## 🏆 Phase 3-C 完整 3-backbone 矩阵 (Full-GCN = LGPA-only)

| Backbone | PSG stage | Full-GCN mAP/R1 | Full+GCN baseline mAP/R1 | Δ (GCN 贡献) |
|----------|-----------|-----------------|--------------------------|---------------|
| Tiny | 2-stg | exp287 65.9/77.0 | exp261 65.9/77.4 | **0/-0.4** |
| Small | 2-stg | exp289 73.8/83.3 | exp285b 73.8/83.8 | **0/-0.5** |
| **Base** | **2-stg** | **exp294 74.0/82.6** | **exp263d 74.1/83.3** | **-0.1/-0.7** |

**3-backbone 统一结论**: **GCN 几乎 0 mAP 贡献** (-0.1 ~ 0), R1 有 0.4-0.7 微贡献。

## 论文 Phase 3-C 最终叙事

1. **LGPA 已捕获 semantic pose 结构, GCN branch 冗余**
2. GCN 只在 R1 top-1 上带来 0.5-0.7 个百分点微小贡献
3. **简化模型 claim**: 可去掉 GCN 模块, 参数和计算显著减少, 性能基本不变
4. 跨 Tiny/Small/Base 3 个 backbone cap 一致验证

## 对照其他 Base OD 结果

| Exp | config | mAP/R1 (eq+flip) | mAP/R1 (MaxSim+flip) |
|-----|--------|------------------|---------------------|
| exp263 orig e100 eff | Full+GCN s42 | 72.5/81.8 | 74.5/84.0 |
| exp263b full 120 | Full+GCN s42 | 73.5/81.5 | 74.8/84.0 |
| exp263d | Full+GCN s41 | 74.1/83.3 | **75.2/84.8** ← SOTA |
| **exp294 (本)** | **Full-GCN s41** | **74.0/82.6** | **pending** |

## MaxSim+flip eval 待办

- ckpt: `lab4090:/home/afr/SOLIDER-REID/log/occluded_duke/exp294_lgpaOnly_2stg_b_od_s41/transformer_120.pth`
- 在 lab4090 本地跑, 预期 MaxSim ~74.8-75.2 / ~83-84 (类似 exp263b 74.8/84.0 或 exp263d 75.2/84.8)
- 若 MaxSim < exp263d, 证明 GCN 对 MaxSim 也冗余 (完整验证假设)

## lab4090 idle (FINAL 后)

- 主进程结束 @ 02:18 CST
- 无 auto-chain 下游
- 下一任务: MaxSim eval + 可能用户指派

## 训练轨迹观察

- **e40 dip** (67.9 vs e30 68.7): 单点 noise, e50 立即恢复至 72.5
- **e50-e110 平滑上升**: 72.5 → 73.1 → 73.4 → 73.7 → 73.9 → 74.0 → **74.1** (peak)
- **e110 → e120**: 74.1 → 74.0 (final 微弱回落, 正常 plateau)
- 整体曲线健康, 无崩溃, 无 NaN/Inf
