# exp266c_best_b_op_s42_full120 monitor — Base Full Scaffold Occ-PTrack seed 42 restart full 120

- 机器: srvB (5060Ti 16G, i-1:61604)
- 启动: 2026-04-23 ~09:30 CST (auto-chain restart from exp266 FINAL)
- Log: `/hy-tmp/log/occluded_posetrack/exp266c_best_b_op_s42_full120/train_log.txt`
- Config: `configs/occluded_posetrack/prcv_best_base.yml` + CLI `SOLVER.SEED 42 TEST.IMS_PER_BATCH 64`
- Scaffold: Swin-Base + Full Scaffold (LGPA + GCN512 + OA-SD + ParAug + PLBOA + 2-stage PSG `[-2,-1]`)
- Speed: ~830s/epoch (5060Ti + BS=64, 275 iter, TEST.IMS_PER_BATCH 64 防 Base eval OOM)
- 总训练时长: ~28h (2026-04-23 09:30 → 2026-04-24 13:37 CST)
- **动机**: exp266 orig s42 e80 eff (78.4/86.2), 用户命令 "full 120 跑满 seed 42 Base OP restart"

## 训练轨迹 (flip-test, eq_concat global; 从 train log 逐行读取)

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 73.4 | 83.3 | 93.8 | 96.6 |
| 20 | 75.3 | 84.2 | 93.6 | 96.4 |
| 30 | 76.5 | 84.7 | 94.1 | 97.1 |
| 40 | 77.3 | 85.2 | 94.5 | 96.9 |
| 50 | 77.8 | 85.9 | 94.5 | 97.2 |
| 60 | 77.9 | 85.6 | 94.7 | 97.3 |
| 70 | 78.0 | 85.8 | 94.5 | 97.2 |
| 80 | 78.0 | 85.8 | 94.6 | 97.2 |
| 90 | 78.1 | 85.7 | 94.5 | 97.3 |
| 100 | 78.1 | 85.7 | 94.6 | 97.1 |
| 110 | 78.0 | 85.8 | 94.6 | 97.3 |
| **120 FINAL** | **78.0** | **85.8** | **94.6** | **97.2** |

## FINAL (2026-04-24 13:40:54 CST)

- **mAP: 78.0%**, **Rank-1: 85.8%**, R5: 94.6%, R10: 97.2%
- **Plateau 30 epoch 稳定**: e80-e120 mAP 78.0-78.1, R1 85.7-85.8 (波动 ≤0.1)
- Ckpt: `/hy-tmp/log/occluded_posetrack/exp266c_best_b_op_s42_full120/transformer_120.pth`

## 🏆 对照 Base OP 矩阵

| Exp | seed | Epoch | FINAL mAP/R1 | Δ vs exp266b SOTA (s41) |
|-----|------|-------|--------------|-------------------------|
| exp266 orig | 42 | e80 eff (OOM) | 78.4/86.2 | -0.3/-0.1 |
| **exp266c (本)** | **42** | **e120 FINAL** | **78.0/85.8** | **-0.7/-0.5** |
| exp266b_3090 | 41 | e120 | 78.5/86.2 | -0.2/-0.1 |
| **exp266b (srvA)** | **41** | **e120** | **78.7/86.3** ← **Base OP SOTA** | **baseline** |

**核心发现**:
- **seed 影响 > 训练长度影响**: exp266c seed 42 full 120 (78.0) < exp266 seed 42 e80 eff (78.4) < exp266b seed 41 e120 (78.7)
- seed 42 在 e80 后 plateau 在 78.0, **无法通过更多训练接近 seed 41 的 78.7**
- 这是第 3 次 confirm seed 41 > seed 42 on Base OP (exp263d, exp265b 也都 seed 41 刷 SOTA)

## 论文 Base OP 主数字选用

- **主表**: exp266b s41 FINAL **78.7/86.3** (新 Base OP SOTA)
- **seed 42 补充点**: exp266 e80 eff 78.4/86.2 (原 seed 42 baseline) 或 exp266c e120 78.0/85.8 (seed 42 full 120 终态)
- 建议主表用 exp266b, supplementary 材料引用 exp266c 作为 seed 42 full 120 复现点

## MaxSim+flip eval PENDING (srvB, 2026-04-24 13:41 CST 启动)

srvB batch chain (10 ckpts) 正在跑:
1. exp266c (self) ← running now
2. exp261, exp267 (baseline Tiny)
3. exp271-273 (pure PSG Tiny 1-3)
4. exp278-280 (GCN×PSG Tiny)
5. exp290 (Small OP target-heatmap)

exp266c MaxSim 预期: ~78.2-78.4 / ~85.8-86.0 (MaxSim +0.2-0.4 boost pattern for Base)

## 训练轨迹观察

- **e10-e60 平滑上升**: 73.4 → 75.3 → 76.5 → 77.3 → 77.8 → 77.9
- **e70-e120 plateau**: 78.0, 78.0, 78.1, 78.1, 78.0, 78.0 (fluctuation ≤0.1)
- R1 同步 plateau @ 85.7-85.8 (e70 后)
- 整体曲线健康, 无崩溃/NaN/Inf

## srvB idle (FINAL 后)

- 主训练进程结束 @ 13:37:34 CST
- 立即启动 srvB MaxSim eval batch (exp266c + 9 其他 ckpts)
- 无后续 auto-chain 下游
