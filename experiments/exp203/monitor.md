# exp203 Swin-Small + GCN+PAA+ROA + SupCon + PLBOA + 3-view 监控

配置: pose_psg_gcn_paa_roa.yml + Small + SupCon T=0.05 + PLBOA 0.7 + 3-view + WITH_CP
对照:
- 4090 PAA (Small, GCN+PAA+ROA, CE): **70.8/81.7**
- exp202b (Small, STD-PR+SupCon+3-view): 69.3/80.2

**目标**: 超过 70.8/81.7！

## 检查点

### [04:40] 检查点 #1

**状态**: 正常
**进度**: Epoch 1/120

| 指标 | 当前值 |
|------|--------|
| supcon | 3.975 |
| id_global | 6.554 |
| tri_global | 15.793 |
| tri_part | 16.912 (GCN part triplet, 比 STD-PR 高) |
| GPU | **6.8GB/24GB** (GCN 很轻量！) |

**观察**: GCN+PAA+SupCon+3-view 成功启动。GPU 仅 6.8GB！
### [04:47] 检查点 #2

**进度**: Epoch 2/120
supcon=3.512, tri_part=7.448 (GCN 的 part triplet 下降快)。
### [04:56] 检查点 #3

ep4/120, ETA 8h07m. Speed 53.6 samples/s.
### [05:07] 检查点 #4

**本地 3-view**: ep7/120, supcon=3.504, tri_part=1.030 (下降快)
**远程 1-view**: ep4/120, speed=111.3, ETA 3h54m
### [05:13] 检查点 #5

### [05:18] 检查点 #6

### [05:21] 检查点 #7 — 远程 ep10

**远程 GCN+PAA+SupCon 1-view ep10**: 36.2/50.6

| 配置 | 架构 | ep10 |
|------|------|------|
| exp202 (STD-PR+SupCon) | STD-PR 6-token | 43.1/56.4 |
| **exp203r (GCN+PAA+SupCon)** | **GCN 1-pooled** | **36.2/50.6** |
| delta | | **-6.9/-5.8** |

**SupCon 在 GCN (1 pooled feat) 上效果远不如 STD-PR (6 per-token)！**
SupCon 需要多个 token 才能发挥。GCN 只有 1 个 pooled skeleton feature。

但 4090 PAA (无 SupCon, CE only) = 70.8。问题是 SupCon 是否能在 GCN 上追加增益？
### [05:23] 检查点 #8 — 本地 ep10

**本地 3-view GCN+PAA+SupCon ep10: 46.9/59.5**

| 配置 | 架构 | 3-view ep10 | 1-view ep10 |
|------|------|------|------|
| exp202b (STD-PR) | 6-token | 56.2/68.9 | 43.1/56.4 |
| **exp203 (GCN+PAA)** | **1-pooled** | **46.9/59.5** | **36.2/50.6** |
| delta | | **-9.3/-9.4** | -6.9/-5.8 |

**STD-PR 在 Small+SupCon 上明显优于 GCN+PAA！**
但 4090 PAA (GCN, CE only) = 70.8 vs exp202b (STD-PR, SupCon) = 69.3。
这说明 GCN+PAA 的 CE 路线天花板更高，但 SupCon 加不上去。
**决策**: 继续观察 GCN+PAA+SupCon 是否能在后期追上
