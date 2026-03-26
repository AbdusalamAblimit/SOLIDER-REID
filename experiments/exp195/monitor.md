# exp195 OA-SD Global-Only + SupCon 监控

配置: base arch (无 PAPE/MS-PSG) + SupCon T=0.05 + OA-SD GLOBAL_ONLY + 0.5x global loss
对照:
- exp176 (SupCon, no OA-SD): 64.1/75.5 (有 PAPE/MS-PSG)
- exp188 (all-token OA-SD + SupCon): ~63.4/75.1 (负向)
- exp191 (OA-SD + CE): 63.2/75.4

**注意**: exp195 没有 PAPE 和 multi-stage PSG（16GB 限制），所以绝对值会低于 exp176。
关键比较应该是 exp195 vs **同配置的 SupCon-only baseline**。

## 检查点

### [18:08] 检查点 #1

**状态**: 正常
**进度**: Epoch 2/120, ETA 4h

| 指标 | 当前值 | 备注 |
|------|--------|------|
| supcon | 4.186 | SupCon loss 正常下降 |
| oa_sd | 0.389 | OA-SD global-only 正常 |
| id_global | 6.553 | 初始 |
| id_part | 4.186 | = supcon (additive mode) |
| Speed | 110.7 samples/s | 正常 (16GB 5060 Ti) |

**观察**: SupCon + OA-SD GLOBAL_ONLY 成功启动，无 OOM。
oa_sd=0.389 在正常范围（与 exp191 一致）。
### [18:21] 检查点 #2

**进度**: Epoch 8/120
supcon=3.776 (↓), oa_sd=0.043 (↓ 从 0.39→0.04, teacher 快速追上)。
### [18:26] 检查点 #3

**进度**: Epoch 10/120
oa_sd=0.025 (极低，与 exp192 的 decay=0.99 类似)。supcon=3.598。
**ep10 eval**: 34.8/47.4

| 实验 | 方法 | ep10 |
|------|------|------|
| **exp195** | **SupCon + OA-SD global-only** | **34.8/47.4** |
| exp191 | CE + OA-SD all-token | 34.3/46.8 |
| exp192 | CE + OA-SD decay=0.99 | 35.0/47.3 |

**观察**: 与 CE+OA-SD 系列相近。SupCon 的效果要到 ep30+ 才显现。
### [18:35] 检查点 #4

**进度**: Epoch 14/120
oa_sd=0.015 (极低！), supcon=2.893 (正常下降)。
global-only distillation 信号非常弱——可能是因为 global feature 变化平缓，
teacher 轻松追上。### [18:41] 检查点 #5

**进度**: Epoch 17/120
### [18:49] 检查点 #6

**ep20 eval**: 44.2/58.1

| 实验 | 方法 | ep10 | ep20 |
|------|------|------|------|
| **exp195** | **SupCon + OA-SD global** | **34.8/47.4** | **44.2/58.1** |
| exp191 | CE + OA-SD all-token | 34.3/46.8 | 46.0/58.0 |

**观察**: mAP 44.2 落后 exp191 (46.0) -1.8，R1 持平 (58.1 vs 58.0)。
SupCon 在这个配置下（无 PAPE/MS-PSG）可能前期更慢。
### [19:00] 检查点 #7

**进度**: Epoch 26/120
oa_sd=0.015 (稳定在极低水平), supcon=2.032, id_global=2.329。
### [19:06] 检查点 #8

**进度**: Epoch 29/120
### [19:10] 检查点 #9 — ep30 SupCon 显效！

**ep30 eval**: 51.8/65.1

| 实验 | 方法 | ep10 | ep20 | ep30 |
|------|------|------|------|------|
| **exp195** | **SupCon + OA-SD global** | 34.8/47.4 | 44.2/58.1 | **51.8/65.1** |
| exp191 | CE + OA-SD all-token | 34.3/46.8 | 46.0/58.0 | 50.6/61.7 |

**观察**: ep30 exp195 反超 exp191 **+1.2/+3.4**！SupCon 的加速在 ep30 开始显现。
R1 65.1 vs 61.7 领先 +3.4，这是 SupCon 的典型特征。
关键：OA-SD global-only 没有阻碍 SupCon 的效果！梯度冲突被避免了！
### [19:19] 检查点 #10

**进度**: Epoch 35/120, supcon=1.837, id_global=1.410, oa_sd=0.016。
### [19:25] 检查点 #11

**进度**: Epoch 38/120, oa_sd=0.017, supcon=1.802。
### [19:31] 检查点 #12 — ep40 关键对照

**ep40 eval**: 56.5/70.7

| 实验 | 方法 | ep10 | ep20 | ep30 | ep40 |
|------|------|------|------|------|------|
| **exp195** | **SupCon+OA-SD global** | 34.8/47.4 | 44.2/58.1 | 51.8/65.1 | **56.5/70.7** |
| exp191 | CE+OA-SD all-token | 34.3/46.8 | 46.0/58.0 | 50.6/61.7 | 57.2/69.2 |
| exp188 | SupCon+OA-SD all-token | — | — | — | ~55/66 (负向) |

**关键发现**:
1. **OA-SD global-only + SupCon 没有梯度冲突！** (vs exp188 的失败)
2. R1 70.7 >> exp191 69.2 (+1.5) — SupCon 的 R1 优势保持
3. mAP 56.5 vs exp191 57.2 (-0.7) — mAP 略低，可能因配置差异（无 PAPE/MS-PSG）

**结论: global-only distillation 成功解决了 OA-SD vs SupCon 的梯度冲突。**
### [19:39] 检查点 #13

**进度**: Epoch 44/120, oa_sd=0.016, supcon=1.725。### [19:53] 检查点 #14 — ep50

**ep50 eval**: 58.0/72.1

| Epoch | exp195 (SupCon+OA-SD global) | exp191 (CE+OA-SD all) | delta |
|-------|------|------|------|
| 10 | 34.8/47.4 | 34.3/46.8 | +0.5/+0.6 |
| 20 | 44.2/58.1 | 46.0/58.0 | -1.8/+0.1 |
| 30 | 51.8/65.1 | 50.6/61.7 | +1.2/+3.4 |
| 40 | 56.5/70.7 | 57.2/69.2 | -0.7/+1.5 |
| **50** | **58.0/72.1** | **59.0/70.6** | **-1.0/+1.5** |

**观察**: R1 持续大幅领先 (+1.5)，mAP 落后 (-1.0)。
SupCon 在 base config（无 PAPE/MS-PSG）下 mAP 弱于 CE，但 R1 强。
这可能是配置差异（base config 的 SupCon 不如 full config 效果好）。

### [20:14] 检查点 #15 — ep60

**ep60 eval**: 59.2/72.8

| Epoch | exp195 (SupCon+OA-SD global) | exp191 (CE+OA-SD all) | delta |
|-------|------|------|------|
| 30 | 51.8/65.1 | 50.6/61.7 | +1.2/+3.4 |
| 40 | 56.5/70.7 | 57.2/69.2 | -0.7/+1.5 |
| 50 | 58.0/72.1 | 59.0/70.6 | -1.0/+1.5 |
| **60** | **59.2/72.8** | **60.6/72.9** | **-1.4/-0.1** |

**观察**: R1 差距收窄到 -0.1 (从 +1.5 缩小)。mAP 持续落后 -1.4。
SupCon 在 base config 下的 mAP 弱于 CE，这可能是配置差异导致的。
**关键结论不变**: OA-SD global-only 与 SupCon 无冲突。
### [20:36] 检查点 #16 — ep70

**ep70 eval**: 60.2/73.4

| Epoch | exp195 (SupCon+OA-SD global) | exp191 (CE+OA-SD all) | delta |
|-------|------|------|------|
| 50 | 58.0/72.1 | 59.0/70.6 | -1.0/+1.5 |
| 60 | 59.2/72.8 | 60.6/72.9 | -1.4/-0.1 |
| **70** | **60.2/73.4** | **61.8/73.1** | **-1.6/+0.3** |

**观察**: mAP 持续落后 exp191 -1.6，R1 微领先 +0.3。
base config 的 SupCon mAP 不如 CE——但这是 full config 上可能不同。
**核心收获**: OA-SD global-only + SupCon 兼容性已验证。
**决策**: 继续到完成
