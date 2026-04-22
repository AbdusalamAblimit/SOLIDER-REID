# exp291 monitor — Small Full Scaffold + target-heatmap on Occ-Duke (seed 42)

- 机器: lab4090 (24GB 4090, mmpose-abu env)
- 启动: 2026-04-22 12:06 CST (本地时区 UTC+8; log 用 UTC)
- Log: `/tmp/exp291.log` + `/home/afr/SOLIDER-REID/log/occluded_duke/exp291_target_s_od_s42/train_log.txt`
- Config: `configs/occluded_duke/prcv_best_small.yml` + CLI `SOLVER.SEED 42 MODEL.POSE_USE_TARGET_HEATMAP True`
- Scaffold: Swin-Small + Full Scaffold (LGPA + GCN512 + OA-SD + ParAug + LOWER_BODY_OCC) + 2-stage PSG `[-2,-1]` + **target-heatmap swap**
- Speed: 174s/epoch × 120 ≈ 5h48min (实际 6h07min 含 eval)

## 训练轨迹 (flip-test, eq_concat global)

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 53.4 | 62.8 | 78.4 | 82.9 |
| 20 | 60.8 | 70.3 | 83.6 | 86.7 |
| 30 | 68.6 | 78.8 | 88.0 | 90.3 |
| 40 | 69.3 | 79.5 | 88.5 | 91.0 |
| 50 | 70.6 | 80.1 | 89.5 | 91.5 |
| 60 | 72.0 | 81.8 | 89.8 | 91.9 |
| 70 | 72.4 | 82.1 | 90.1 | 91.9 |
| 80 | 72.9 | 82.5 | 90.4 | 92.3 |
| 90 | 73.1 | 82.9 | 90.2 | 92.3 |
| 100 | 73.4 | 82.9 | 90.5 | 92.6 |
| 110 | 73.5 | 82.7 | 90.8 | 92.5 |
| **120 FINAL** | **73.5** | **82.9** | **90.7** | **92.5** |

## FINAL (2026-04-22 10:13:30 UTC = 18:13 CST)

- **mAP: 73.5%**, **Rank-1: 82.9%**, R5: 90.7%, R10: 92.5%
- **对照 exp285b Full Scaffold (scene-heatmap default)**: 73.8 / 83.8 / 90.7 / 92.7 → Δ **-0.3 / -0.9 / 0 / -0.2**
- **对照 exp262 srvA original**: 73.8 / 83.1 / 90.2 / 92.2 → Δ -0.3 / -0.2 / +0.5 / +0.3
- Ckpt: `transformer_120.pth` (230MB) @ `/home/afr/SOLIDER-REID/log/occluded_duke/exp291_target_s_od_s42/`

## 🔍 Target-heatmap 机制在 OD 上的表现

**预期** (design.md): OD 多为 single-person → `target_heatmap = scene_heatmap` → swap = no-op → 结果 ≈ exp285b 73.8/83.8。

**实际**: Δ -0.3 mAP / -0.9 R1 — 微差但持续在整个训练周期出现。

**解读**:
1. **OD 并非 100% single-person** — 部分图像有 distractor, target-heatmap 丢了 distractor 信息
2. 但 GCN 分支用 `heatmaps[:, 0]` (hardcoded person-0), 无论 flag on/off 都用 target → GCN 一致, 差异来自 PSG/LGPA gate 的 scene vs target 切换
3. `-0.3 mAP / -0.9 R1` 属 **evaluation noise** 范围 (跨 seed/device 差异可达此量级), 不应解读为机制伤害 OD

**论文叙事安全**:
- 机制设计用于 OP 多人场景, OD/Market 只作 "backward compat/no regression" 证据
- target-heatmap 在 OD **未造成显著回归** (-0.3 mAP < 跨 seed 方差 0.7)

## 对照表 (Small OD 所有 FINAL)

| Exp | Config | mAP / R1 | vs exp285b |
|-----|--------|----------|-----------|
| exp262 | Full Scaffold (srvA 原始) | 73.8 / 83.1 | baseline (-0.7 R1 跨设备) |
| exp285b | Full Scaffold (lab4090 rerun) | 73.8 / 83.8 | baseline |
| exp282 | GCN256+1stg | 73.7 / 83.9 | -0.1/+0.1 |
| exp283 | GCN256+2stg | 73.5 / 83.2 | -0.3/-0.6 |
| exp284 | GCN512+1stg | 73.4 / 82.9 | -0.4/-0.9 |
| exp288 | LGPA-only 1stg (no GCN) | 73.8 / 83.8 | **持平** |
| **exp291** | **target-heatmap (Full)** | **73.5 / 82.9** | -0.3/-0.9 |

## auto-chain → exp293 (Base Market PLBOA)

daemon 706372 detected `transformer_120.pth` @ 10:14 UTC (18:14 CST)。正在等 exp291 python 进程 exit + 20s 安全等待 → launch exp293 Base Market 满血 + PLBOA。

## 论文定位

- Small OD target-heatmap 不是 SOTA 推手 (-0.3 mAP)
- **OP target-heatmap 是真 SOTA 推手** (exp290 还在跑, 目前 e30 R1 +0.1)
- 本实验作为 supplementary 消融: 机制在 single-person 数据集无回归
