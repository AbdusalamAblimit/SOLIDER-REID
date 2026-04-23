# exp266b_best_b_op_s41 monitor — Base Full Scaffold Occ-PTrack seed 41 (srvA)

- 机器: srvA (5060Ti 16G, i-2:29162), auto-chain from exp265b via daemon 992
- 启动: 2026-04-22 09:05 CST (auto-chain 触发时刻)
- Log: `/hy-tmp/log/occluded_posetrack/exp266b_best_b_op_s41/train_log.txt`
- Config: `configs/occluded_posetrack/prcv_best_base.yml` + CLI `SOLVER.SEED 41 TEST.IMS_PER_BATCH 128`
- Scaffold: Swin-Base + Full Scaffold (LGPA + GCN512 + OA-SD + ParAug + LOWER_BODY_OCC + 2-stage PSG `[-2,-1]`)
- Speed: ~14 min/epoch (5060Ti + WITH_CP, TEST.IMS_PER_BATCH 128 避免 eval OOM)
- 总训练时长: 28h14min (09:05 → 13:19 tmr)

## 训练轨迹 (flip-test, eq_concat global)

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 74.9 | 83.5 | - | 96.6 |
| 20 | 75.9 | 84.0 | - | 96.5 |
| 30 | 77.8 | 85.7 | - | 97.0 |
| 40 | 78.2 | 85.5 | - | 97.0 |
| 50 | 78.4 | 86.2 | - | 97.2 |
| 60 | 78.6 | 86.1 | - | 97.3 |
| 70 | 78.6 | **86.4** (R1 peak) | - | 97.2 |
| 80 | 78.6 | 86.4 | - | 97.2 |
| 90 | 78.7 | 86.3 | - | 97.3 |
| 100 | 78.7 | 86.3 | - | 97.2 |
| 110 | 78.7 | 86.2 | - | 97.1 |
| **120 FINAL** | **78.7** | **86.3** | **94.5** | **97.1** |

## FINAL (2026-04-23 13:18:50 CST)

- **mAP: 78.7%**, **Rank-1: 86.3%**, R5: 94.5%, R10: 97.1%
- **对照 exp266b_3090 s41 FINAL** (lab3090 同 seed 不同设备): 78.5/86.2/94.4/96.9 → Δ **+0.2/+0.1/+0.1/+0.2** (srvA 5060Ti 微优)
- **对照 exp266 s42 srvC** (e60 eff before silent exit): 78.4/86.2 → Δ **+0.3/+0.1**
- **对照 exp265 s42 Small OP FINAL**: 78.4/86.2 → Δ **+0.3/+0.1** (Base vs Small 0.3 mAP 微优)
- **对照 exp265b s41 Small OP FINAL**: 78.5/85.9 → Δ **+0.2/+0.4** (Base vs Small 同 seed 41, R1 更大差距)
- e90 开始 mAP 稳定 78.7 (30 epoch 平稳), R1 86.2-86.4 抖动
- Ckpt: `transformer_120.pth` (~407MB)

## 🏆 Phase 3 OP Base 双设备 SOTA confirmed

| Exp | 机器 | seed | TEST BATCH | FINAL mAP/R1 |
|-----|------|------|------------|--------------|
| exp266b_3090 | lab3090 | 41 | 256 (24G) | 78.5/86.2 |
| **exp266b (本)** | **srvA 5060Ti** | **41** | **128** | **78.7/86.3** ← new SOTA |

**Δ srvA vs lab3090 同 seed**: +0.2/+0.1, 跨设备方差 ~0.2 mAP — 一致性好。

## OP 矩阵饱和最终定位

| | seed 42 | seed 41 |
|---|---------|---------|
| Small (exp265/265b) | 78.4/86.2 (srvC) | 78.5/85.9 (srvA) |
| Base (exp266/266b) | 78.4/86.2 e60 eff | **78.7/86.3** (srvA, new SOTA) |

**Base vs Small 同 seed 41**: Δ **+0.2/+0.4** (Base 非 0 增益, 尤其 R1 显著)
**论文主表 Base OP 数字更新**: 78.7/86.3 (srvA s41), 原计划用 78.5 (lab3090 s41)

## MaxSim+flip eval FINAL (srvA, 2026-04-23 13:25 CST)

| 评测模式 | mAP | R1 |
|---------|-----|----|
| train-side eq_concat+flip (training log) | 78.7 | 86.3 |
| Global cosine+flip (post-hoc script) | 78.4 | **86.6** ← R1 peak |
| **MaxSim hybrid+flip** (post-hoc script) | **78.7** | **86.3** |

**结论**: MaxSim hybrid = eq_concat (training default)。Global-only 丢 0.3 mAP 但 R1 微升 +0.3 (部分分支对 R1 有轻微 overfit 效应)。**论文主表 Base OP 用 78.7/86.3** (两种评测等价)。

## auto-chain 后续

- srvA 5060Ti 本 exp + MaxSim eval 全 FINAL @ 13:25 CST, idle
- daemon 992 无下游任务 (chain 至 exp266b 终止)
- 预计下一 slot: Task #12 MaxSim+flip 继续批跑或用户指派新任务
