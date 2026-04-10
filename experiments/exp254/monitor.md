# exp254 Tiny 2-Stage PSG (Stage2+3, 无 PAA) + LGPA-D+GCN 监控

配置: Swin-Tiny + PSG Stage2+3 (无 PAA) + LGPA-D+GCN+OA-SD+PLBOA + WITH_CP
环境: 远程 5060Ti
对照: exp246b (1-stage) 65.5/77.2 | exp251 (2-stage+PAA) 65.2/76.2 | exp253 (3-stage) 65.1/76.2

## 检查点

### [14:38] 检查点 #1 — 启动成功

ep1 iter 80, 5516 MiB (< exp251 5542, 无 PAA 确认). 训练正常。

### [14:38] 检查点 #2 (cron) — ep1, 刚启动

ep10 eval ~15:30。训练正常。

### [15:29/15:38] 检查点 #4 — ⭐ ep10 = 42.7/56.3 + ep12

**ep10**: mAP=42.7%, R1=56.3%

| Exp | PSG stages | PAA | ep10 mAP/R1 |
|-----|-----------|-----|-------------|
| exp246b | S3 | No | 42.2/55.4 |
| **exp254** | **S2+S3** | **No** | **42.7/56.3** |
| exp251 | S2+S3 | Yes | 42.0/55.4 |
| exp253 | S1+S2+S3 | No | 42.0/56.4 |

ep10 微高于其他 (+0.5~0.7 mAP)。继续观察。
**决策**: 继续

### [16:22/16:38] 检查点 #6 — ep20 = 51.4/63.6 + ep24

**ep20**: mAP=51.4%, R1=63.6% (vs exp246b 52.2/65.9 = -0.8/-2.3)
ep10 was +0.5, ep20 is -0.8. Normal variance across PSG variants.
**决策**: 继续

### [17:15/17:38] 检查点 #8 — ⭐ ep30 = 58.0/70.5 + ep35

**ep30**: mAP=58.0%, R1=70.5% (vs exp246b 57.5/69.9 = **+0.5/+0.6**)
所有 multi-stage 变体 ep30 都正向: +0.5~1.5。
**决策**: 继续

### [18:08] 检查点 #10 — ⭐ ep40 = 62.0/74.9 — best at ep40!

**ep40**: mAP=62.0%, R1=74.9% (vs exp246b 61.3/73.9 = **+0.7/+1.0**)
**ep40 所有变体最高!** 2-stage PSG (no PAA) 在 ep40 表现最好。
**决策**: 继续，有可能 final 也略好

### [19:01/19:08] 检查点 #12 — ⭐ ep50 = 63.1/75.5 — still leading!

**ep50**: mAP=63.1%, R1=75.5% (vs exp246b 62.8/74.5 = **+0.3/+1.0**)
**ep40+ep50 都是所有变体最高!** 2-stage PSG (no PAA) 可能是最优。
**决策**: 继续

### [19:54/20:08] 检查点 #14 — ep60 = 63.9/75.6 + ep63

**ep60**: mAP=63.9%, R1=75.6% (vs exp246b 64.0/75.2 = -0.1/**+0.4**)
R1 仍领先! mAP 趋同。~5h to final。

### [20:46/21:08] 检查点 #16 — ⭐ ep70 = 64.5/76.6 + ep75

**ep70**: mAP=64.5%, R1=76.6% (vs exp246b 64.6/76.3 = -0.1/**+0.3**)
**R1 76.6 是所有 PSG 变体 ep70 最高!** 持续领先。

### [21:39] 检查点 #18 — ⭐⭐ ep80 = 65.1/76.9 — both leading!

**ep80**: mAP=65.1%, R1=76.9% (vs exp246b 65.0/76.2 = **+0.1/+0.7**)
**mAP 首次超越 baseline! R1 76.9 新高!** 2-stage PSG 是最优 multi-stage 配置。
ETA ~3h。

### [22:32/22:38] 检查点 #20 — ep90 = 64.9/76.9 + ep92

**ep90**: mAP=64.9%, R1=76.9% (vs exp246b 65.3/76.7 = -0.4/**+0.2**)
R1 仍最高! mAP plateau。~2.5h to final。

### [14:47] exp254b (Small 2-stage PSG, no PAA) 本地启动

本地 3090 空闲，启动 Small 版本验证 exp254 的优势。
对照: exp249 (Small 1-stage PSG): 71.9/81.8, MaxSim 73.3/83.2
GPU: 8838/24576 MiB

### [23:08] 检查点 #21 (cron) — exp254 ep97, exp254b ep5

**exp254 (Tiny)**: ep97, ep100 eval ~15 min. Final ~1.5h.
**exp254b (Small)**: ep5, healthy. ep10 eval ~15:50.

### [23:26/23:38] 检查点 #22 — ⭐ ep100 = 65.3/77.0!

**exp254 ep100**: mAP=65.3%, R1=77.0% (vs exp246b 65.4/76.8 = -0.1/**+0.2**)
**R1 77.0 新高!** mAP 也涨到 65.3 (接近 baseline 65.4)。
**exp254b (Small)**: ep10, eval imminent.
Final ~1h。

### [15:40] exp254b (Small) ep10 = 50.9/60.8

vs exp249 51.1/61.7 = -0.2/-0.9. Baseline-level at ep10.

### [00:08] 检查点 #23 (cron) — exp254 ep108, exp254b ep16

**exp254**: ep108, ETA 1h3m. ep110 eval ~8 min.
**exp254b**: ep16, healthy.

### [00:20] 检查点 #24 — ⭐ ep110 = 65.5/77.0!

**ep110**: mAP=65.5%, R1=77.0% (= exp246b 65.5/76.9 = **0.0/+0.1**)
**追平 baseline mAP! R1 仍领先!** Final ~30 min。

### [00:30] 检查点 #26 — exp254 ep112 ETA 41min. exp254b ep20 eval imminent.

准备 exp255 (Small GCN 512 + 2-stage PSG)，review agent 跑着。exp254 完成后立即启动。

### [16:33] exp254b (Small) ep20 = 60.9/72.1

vs exp249 60.9/73.2 = 0.0/-1.1. mAP 持平, R1 略低。

### [01:14] 🎉 exp254 FINAL!

**FINAL: mAP=65.5%, R1=76.8%, R5=87.6%, R10=90.0%**

| 方法 | mAP | R1 | vs exp246b |
|------|-----|----|------------|
| exp246b (1-stage) | 65.5 | 77.2 | — |
| **exp254 (2-stage)** | **65.5** | **76.8** | **0.0/-0.4** |
| exp251 (2-stage+PAA) | 65.2 | 76.2 | -0.3/-1.0 |
| exp253 (3-stage) | 65.1 | 76.2 | -0.4/-1.0 |

**2-stage PSG 是最优 multi-stage: mAP = baseline, R1 最接近!**
PAA 反而有害 (exp254 > exp251)。

### [01:18] MaxSim = 66.2/78.1 + exp255 启动!

**MaxSim**: mAP=66.2%, R1=78.1% (**R1 78.1 是所有 Tiny 最高!**)
exp255 (Small GCN 512 + 2-stage PSG) 已在远程启动。

### [17:27/01:38] 检查点 #36 — exp254b ep30 = 64.8/75.4! exp255 ep3.

**exp254b (Small 2-stage PSG) ep30**: mAP=64.8%, R1=75.4%
**vs exp249 ep30**: 63.6/74.2 = **+1.2/+1.2!** 2-stage PSG 在 Small 上 ep30 也大幅正向!
**exp255** (Small GCN 512): ep3, 刚启动。

### [18:20/02:37] 检查点 #38 — exp254b ep40 = 68.0/78.3. exp255 ep10 eval imminent.

**exp254b ep40**: 68.0/78.3 (vs exp249 68.0/78.7 = 0.0/-0.4). Baseline level.

### [19:14/03:37] 检查点 #40 — exp254b ep50 = 69.5/79.8, exp255 ep18

**exp254b ep50**: 69.5/79.8 (vs exp249 69.4/79.4 = +0.1/+0.4). Baseline.

### [20:07/04:08] 检查点 #41 — exp254b ep60 = 70.5/80.5, exp255 ep20 = 62.2/74.3

**exp254b ep60**: 70.5/80.5 (vs exp249 70.2/80.7 = +0.3/-0.2). Baseline.
**exp255 ep20**: 62.2/74.3 (**+1.3/+1.1 vs baseline!**) GCN 512 仍领先!

### [21:01/05:08] 检查点 #43 — exp254b ep70 = 70.9/81.3, exp255 ep29

**exp254b ep70**: 70.9/81.3 (vs exp249 70.9/81.6 = 0.0/-0.3). Baseline.

### [06:08/22:08] 检查点 #45 — exp254b ep80 = 71.4/80.7. exp255 ep36.

exp254b 趋同 baseline.

### [01:31] 🎉 exp254b (Small 2-stage PSG) FINAL!

**FINAL**: mAP=71.7%, R1=81.0% (vs exp249 71.9/81.8 = -0.2/-0.8). Baseline.
2-stage PSG 在 Small 上无额外收益 (与 Tiny 一致)。
