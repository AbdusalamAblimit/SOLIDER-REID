# exp209 Small + STD-PR + CE + OA-SD (远程 1-view) 监控

配置: Swin-Small + STD-PR + CE + OA-SD (no SupCon)
对照: exp202 (STD-PR+SupCon 1v Small): 67.9/79.5, exp206 (GCN+PAA+OA-SD): 70.5/82.3

## 检查点

### [16:34] 检查点 #1

**状态**: 正常
**进度**: Epoch 2/120, ETA 6h49m

| 指标 | 当前值 |
|------|--------|
| oa_sd | 0.785 |
| id_global | 6.554 |
| tri_global | 10.362 |
| str_token_norm | 90.6 |
| str_num_parts | 6.0 |
| GPU | 4.7GB/16GB |

STD-PR+CE+OA-SD 启动正常。oa_sd=0.785 (比 GCN 路线的 0.48 高)。
id_global 还没开始下降——warmup 阶段正常。
**决策**: 继续

### [16:47] 检查点 #2

**状态**: 正常
**进度**: Epoch 4/120

| 指标 | 当前值 |
|------|--------|
| oa_sd | 0.308 |
| id_global | 6.542 (开始微降) |
| tri_global | 1.809 |
| Acc | 0.008 |
| GPU | 4.7GB/16GB |

id_global 从 6.554→6.542 开始下降。oa_sd 从 0.785→0.308 快速下降（teacher 在追 student）。
Acc 还很低但在 warmup 阶段正常。
**决策**: 继续

### [17:23] 检查点 #3

**状态**: 正常
**进度**: Epoch 15/120, ETA 5h59m

| 指标 | 当前值 |
|------|--------|
| id_global | 5.735 (快速下降) |
| Acc | 0.163 |
| oa_sd | 0.017 (极低) |
| tri_global | 0.665 |
| str_token_norm | 103.2 |

学习正常。id_global 从 6.554→5.735 显著下降。Acc=0.163 正常。
oa_sd=0.017 极低——teacher 几乎跟上 student。
ep20 eval ~17min。
**决策**: 等 eval

### [17:10] 检查点 #4 — ep10 ⚠️

**ep10: 32.7/45.2** — 远低于 GCN+PAA+OA-SD (exp206 ep10: 47.9/60.3)!

id_global=6.321 (几乎未下降), Acc=0.097。STD-PR 的 7 个 classifier 在 CE 下收敛更慢。
可能原因：STD-PR + CE (无 SupCon) 不是好的组合——per-token CE 分散了梯度。
**决策**: 等 ep20 eval 确认是否只是慢收敛

### [17:42] 检查点 #5

ep20 mid. id_global=4.894 (终于下降了), Acc=0.153。
学习在加速。ep20 eval ~3min。
**决策**: 等 eval

### [17:45] 检查点 #6 — ep20

**ep20: 48.3/61.6** (vs exp206 ep20: 56.6/68.3 = -8.3/-6.7)

| Epoch | exp209 STD-PR+CE+OA-SD | exp206 GCN+PAA+CE+OA-SD |
|-------|------|------|
| 10 | 32.7/45.2 | 47.9/60.3 |
| 20 | 48.3/61.6 | 56.6/68.3 |

STD-PR+CE 严重落后 GCN+PAA+CE。ep10→ep20 涨了 +15.6/+16.4 (很快)，但仍落后 8.3%。
增长率有追赶趋势，但差距太大，final 预计 ~62-65% (vs exp206 70.5)。
**STD-PR 在 CE 路线下不如 GCN+PAA。**

**决策**: 继续到 ep30 确认趋势。如果 ep30 仍落后 >5%，终止。

### [17:53] 检查点 #7

ep23. id_global=4.474, Acc=0.212. 学习继续。ep30 eval ~23min。
**决策**: 等 ep30 eval

### [17:57] 检查点 #8

ep24. id_global=4.215, Acc=0.283. ep30 eval ~18min。
**决策**: 等 ep30 eval

### [18:01] 检查点 #9

ep25. id_global=3.920, Acc=0.358. ep30 eval ~14min。
**决策**: 等 ep30 eval

### [18:16] 检查点 #10

ep29 done. id_global=3.169, Acc=0.541. ep30 eval 即将开始。
**决策**: 等 eval，然后决定是否终止

### [18:22] 检查点 #11 — ep30 终止

**ep30: 56.0/69.3** (vs exp206 ep30: 60.8/72.1 = -4.8/-2.8)

| Epoch | exp209 STD-PR+CE+OA-SD | exp206 GCN+PAA+CE+OA-SD |
|-------|------|------|
| 10 | 32.7/45.2 | 47.9/60.3 |
| 20 | 48.3/61.6 | 56.6/68.3 |
| 30 | 56.0/69.3 | 60.8/72.1 |

**结论**: STD-PR+CE+OA-SD 在所有 epoch 持续落后 GCN+PAA+CE+OA-SD ~5%。
STD-PR 的 6 个 per-token classifier 在 CE 路线下分散梯度，不如 GCN 的单 pooled classifier。
**STD-PR 需要 SupCon 才能发挥，但 SupCon 与 OA-SD 互斥。**

**实验终止。远程 GPU 转给 exp210 (GCN+PAA+CE+OA-SD+PKC)。**
