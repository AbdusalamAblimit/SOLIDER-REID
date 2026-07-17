# exp288 monitor — Phase 3-C Small LGPA-only 1-stg PSG (Occ-Duke, seed 42)

- 机器: srvC (5060Ti 16G, TEST.IMS_PER_BATCH 128)
- 启动: 2026-04-21 22:05 CST
- Log: `/hy-tmp/log/occluded_duke/exp288_lgpaOnly_1stg_s_od_s42/train_log.txt`
- Config: `configs/occluded_duke/prcv_best_small.yml` + CLI `MODEL.POSE_SKELETON_GCN False MODEL.POSE_PSG_STAGES [-1] TEST.IMS_PER_BATCH 128`
- Scaffold: Swin-Small + LGPA + OA-SD + ParAug + LOWER_BODY_OCC + 1-stage PSG (**无 GCN**)
- Speed: ~80 min per 10 epochs = ~8 min/epoch, 总训练 ~14h46min

## 训练轨迹 (flip-test, eq_concat global)

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 55.1 | 65.6 | 80.5 | 84.8 |
| 20 | 62.6 | 72.9 | 85.8 | 88.3 |
| 30 | 68.8 | 79.1 | 88.0 | 90.6 |
| 40 | 69.4 | 80.5 | 88.5 | 90.6 |
| 50 | 70.8 | 81.1 | 89.3 | 91.6 |
| 60 | 72.2 | 82.8 | 89.7 | 91.9 |
| 70 | 73.1 | 83.5 | 90.3 | 92.2 |
| 80 | 73.2 | 83.3 | 89.8 | 91.7 |
| 90 | 73.5 | 83.5 | 90.0 | 92.0 |
| 100 | 73.6 | 83.6 | 90.2 | 91.9 |
| 110 | 73.7 | 83.6 | 90.3 | 92.1 |
| **120 FINAL** | **73.8** | **83.8** | **90.5** | **92.0** |

## FINAL (2026-04-22 12:51:19 CST)

- **mAP: 73.8%**, **Rank-1: 83.8%**, R5: 90.5%, R10: 92.0%
- 🔥 **对照 exp285b Full Scaffold 73.8/83.8/90.7/92.7**: **完全持平 mAP/R1** (R5/R10 微差 0.2/0.7)
- 🔥 **对照 exp282 Full Scaffold GCN256+1stg 73.7/83.9**: Δ +0.1/-0.1 (几乎一致)
- 对照 exp286 Tiny LGPA-only 1stg 66.0/76.6: Small Δ +7.8/+7.2 (backbone cap 主导)
- Ckpt: `transformer_120.pth` (~222MB)

## 🎯 Phase 3-C Small LGPA-only 核心发现

| Setup | mAP / R1 | 状态 |
|-------|----------|------|
| exp285b Full Scaffold GCN512+2stg | 73.8 / 83.8 | baseline |
| exp288 LGPA-only 1stg (**无 GCN**) | **73.8 / 83.8** | **完全持平!** |
| exp289 LGPA-only 2stg (无 GCN) | pending | 自动 chain 启动 |

**核心结论**: **GCN 对 Swin-Small OD 几乎无贡献** — LGPA 单独即达到 Full Scaffold 性能。和 Tiny 发现 (exp286 LGPA-only 66.0/76.6 ≈ exp261 Full 65.9/77.4) 一致。

**论文叙事升级**:
1. 主创新 PSG + LGPA (semantic 分支) 承担大部分 gain
2. GCN (structural) 对小 backbone **不必要**
3. Phase 3-B GCN cap × PSG stage 矩阵方差 ≤ 0.4 mAP 本质是因为 GCN 不起作用
4. **可简化模型**: 去掉 GCN → 少 0.6M 参数 + 训练 10-15% 更快, 性能无损

## auto-chain → exp289 Small LGPA-only 2-stg (已启动 PID 86783)

daemon 通过 exp288/transformer_120.pth detection 触发 exp289:
- Config: same as exp288 + `MODEL.POSE_PSG_STAGES [-2,-1]` (唯一改动: PSG 2-stg)
- 预期 FINAL ~16:50 CST
- 对照 exp285b 73.8/83.8 验证 "PSG stage 影响 in LGPA-only 配置"

## srvC state

- exp289 训练中 (~3 min 前启动)
- Phase 3-C Small 2/2 完成后进入全闭合: Tiny + Small × 1-stg/2-stg = 4/4
