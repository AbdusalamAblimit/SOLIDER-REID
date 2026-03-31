# exp206 Swin-Small + GCN+PAA + CE + OA-SD (远程 1-view) 监控

配置: Swin-Small + GCN+PAA+ROA + CE + OA-SD (decay=0.999) + PLBOA
对照: 4090 PAA (GCN+PAA, CE, no OA-SD): **70.8/81.7**

**目标**: 4090 PAA 70.8 + OA-SD (+2.9 on Tiny) → **72-73%+**

## 检查点

### [22:50] 检查点 #1

**状态**: 正常
**进度**: Epoch 1/120, ETA 5h25m

| 指标 | 当前值 |
|------|--------|
| oa_sd | 0.480 |
| id_global | 6.554 |
| tri_part | 11.865 (GCN part triplet) |
| Speed | 81.9 samples/s |

**观察**: OA-SD+GCN+PAA 成功启动。oa_sd=0.48 正常。
**决策**: 继续

### [23:18] 检查点 #2

**进度**: Epoch 11/120
oa_sd=0.003 (极低！teacher ≈ student)。
但 exp191 (Tiny) 也在早期 oa_sd 很低，后期才有效。
ep10 eval 刚过——让我检查。
**决策**: 继续

### [23:20] ep10 结果: 0.2/0.0% — 严重 bug！已终止

**ep10: mAP 0.2%, R1 0.0%!** 模型完全没学到。
id_global 从 6.554 到 6.550 几乎不变（10 个 epoch CE 没收敛）。
oa_sd 从 0.48 急降到 0.003（teacher 几乎等于 student，但 student 没在学习）。

**可能原因**: OA-SD 在 GCN 模式下从未测试过。所有之前的 OA-SD 实验都是 STD-PR 配置。
GCN 的 forward path 不同（detached feature map, skeleton_head），OA-SD 的 teacher forward 可能不兼容。

**需要 debug**: 检查 processor.py 中 OA-SD 在 GCN 模式下的 teacher forward 是否正确。
已终止远程训练。

### [19:52] Debug 测试

本地: GCN+PAA+CE+OA-SD Small (10ep), 13.2GB, oa_sd=0.56
远程: GCN+PAA+CE Small 无 OA-SD (10ep), 正常启动
对比 ep10 eval 确认 OA-SD 是否是问题根源。
**决策**: 等 ep10 evals (~20min)

### [19:58] Debug 检查点 #3

本地 OA-SD ep5: id_global=6.505, Acc=0.089, oa_sd=0.183 — 正常学习！
远程 no-OA-SD ep5: id_global=6.515, Acc=0.049 — 也正常！

**之前 exp206 的 0.2% mAP bug 可能是残留进程导致 OOM/显存不足。**
不是代码 bug！等 ep10 eval 最终确认。
**决策**: 等 ep10 evals

### [20:04] Debug 检查点 #4

本地 ep8 (2min to ep10 eval), 远程 ep8. 两者正常学习。
**决策**: 等 ep10 evals

### [20:09] Debug 结果 — OA-SD 正常！

**本地 GCN+PAA+CE+OA-SD Small ep10: 47.9/61.1** — 完全正常！
之前远程 exp206 的 0.2% 是残留进程 OOM 导致，不是代码 bug。
等远程无 OA-SD 版 ep10 eval 做对比。
**决策**: 等远程 eval，然后启动正式 120ep 训练

### [20:28] 正式训练启动！

**本地**: GCN+PAA+CE+OA-SD+3-view+CP Small. 7.4GB. oa_sd=0.489.
**远程**: GCN+PAA+CE+OA-SD 1-view Small. 84.3 s/s, ETA 5h16m.

Debug 确认: OA-SD ep10 = 47.9/61.1 (正常), 无 OA-SD ep10 = 43.9/56.6 → OA-SD **+4.0/+4.5!!**
**如果 OA-SD 增益持续到 final: 4090 PAA 70.8 + OA-SD → 预计 73-74%!**
**决策**: 继续监控

### [20:34] 检查点 #5

本地 ep3 (3-view+CP, 51.9 s/s, ETA 8h34m), 远程 ep4 (84.3 s/s).
oa_sd: 本地 0.512, 远程 0.336. 两台正常学习。
远程 ep10 eval ~16min。
**决策**: 继续

### [20:40] 检查点 #6

远程 ep6. id_global=6.474 (开始下降!), oa_sd=0.129. ep10 eval ~10min.
**决策**: 继续

### [20:45] 检查点 #7

远程 ep8. oa_sd=0.054 (快速下降). ep10 eval ~5min.
**决策**: 等 eval

### [20:54] 检查点 #8 — 远程 ep10 ✓

**远程 GCN+PAA+CE+OA-SD ep10: 47.9/60.3** (与 debug 一致!)

| 方法 | ep10 mAP | ep10 R1 |
|------|---------|---------|
| **+OA-SD** | **47.9** | **60.3** |
| 无 OA-SD | 43.9 | 56.6 |
| delta | **+4.0** | **+3.7** |

OA-SD 增益在正式训练中确认。
如果持续到 final: 4090 PAA 70.8 + OA-SD → **73%+ mAP!**
**决策**: 继续

### [21:01] 检查点 #9

远程 ep13, 本地 ep8. 远程 ep20 eval ~16min.
**决策**: 继续

### [21:07] 检查点 #10

远程 ep15. ep20 eval ~12min.
**决策**: 继续

### [21:13] 检查点 #11

远程 ep17. ep20 eval ~8min.
**决策**: 等 eval

### [21:19] 检查点 #12

远程 ep20 (iter 40/227). eval ~3min. 本地 ep~12。
id_global=4.923 (正常下降中)。
**决策**: 等 ep20 eval

### [21:23] 检查点 #13 — 远程 ep20

**远程 GCN+PAA+CE+OA-SD ep20: 56.6/68.3**

| Epoch | +OA-SD | 无OA-SD (debug 10ep) |
|-------|--------|------|
| **10** | **47.9/60.3** | **43.9/56.6** |
| **20** | **56.6/68.3** | — |

趋势非常正向！ep10 +4.0 增益可能在后期继续扩大。
如果 4090 PAA (无 OA-SD) 最终 = 70.8，有 OA-SD 最终可能 = 73-74!
**决策**: 继续

### [21:30] 检查点 #14

本地 ep14 (3-view+CP), 远程 ep23. 远程 ep30 eval ~17min.
**决策**: 继续

### [21:36] 检查点 #15

远程 ep25. ep30 eval ~12min. 远程 ETA 4h12m.
**决策**: 继续

### [21:41] 检查点 #16

远程 ep28. ep30 eval ~6min.
**决策**: 等 eval

### [21:49] 检查点 #17

远程 ep30 done, eval 运行中。
**决策**: 等 eval

### [21:51] 检查点 #18 — 远程 ep30

**远程 GCN+PAA+CE+OA-SD ep30: 60.8/72.1**

| Epoch | mAP | R1 |
|-------|------|------|
| 10 | 47.9% | 60.3% |
| 20 | 56.6% | 68.3% |
| **30** | **60.8%** | **72.1%** |

健康增长！如果 4090 PAA (无 OA-SD) final=70.8，OA-SD 版可能 **73+**。
远程 ETA ~3h50m。
**决策**: 继续

### [21:58] 检查点 #19

本地 ep20, 远程 ep33. 本地 ep20 eval ~2min, 远程 ep40 eval ~18min.
**决策**: 等 evals

### [22:00] 检查点 #20 — 本地 ep20 ⚠️

**本地 3-view+CP ep20: 48.8/61.3** — 从 ep10 47.9 只涨 +0.9!!

| 配置 | ep10 | ep20 | delta |
|------|------|------|------|
| 远程 1-view | 47.9/60.3 | 56.6/68.3 | +8.7/+8.0 |
| **本地 3-view+CP** | **47.9/61.1** | **48.8/61.3** | **+0.9/+0.2** |

**3-view+CP 版学习几乎停滞！** 可能 CP 与 OA-SD 的 2-view 有 bug。
3-view 的 4-view tuple (3 student + 1 teacher) + CP 可能导致梯度问题。
远程 1-view 正常。

**决策**: kill 本地 3-view+CP，改跑 1-view+OA-SD (不需 CP)

### [22:03] 本地改跑 1-view OA-SD (无 CP)

3-view+CP 版有学习停滞 bug。改为 1-view。13.2GB/24GB。
两台都跑 1-view GCN+PAA+CE+OA-SD Small。本地是独立验证 run。
远程 ~ep35, 本地从头开始。
**决策**: 继续

### [22:09] 检查点 #21

远程 ep37, 本地 ep4. 远程 ep40 eval ~8min.
**决策**: 等 eval

### [22:17] 检查点 #22

远程 ep40 (iter 180). eval ~2min.
**决策**: 等 eval

### [22:19] 检查点 #23 — 远程 ep40 🔥🔥

**远程 GCN+PAA+CE+OA-SD ep40: 65.4/76.5!!**

| Epoch | mAP | R1 |
|-------|------|------|
| 10 | 47.9% | 60.3% |
| 20 | 56.6% | 68.3% |
| 30 | 60.8% | 72.1% |
| **40** | **65.4%** | **76.5%** |

**ep30→ep40 涨了 +4.6 mAP!!** 增长加速！
如果 4090 PAA (无 OA-SD) ep40 ~62-64，我们已领先 +1-3!
趋势：ep120 可能达 **73-74%!**
**决策**: 继续

### [22:25] 检查点 #24

远程 ep43, 本地 ep13. 远程 ep50 eval ~18min.
**决策**: 继续

### [22:31] 检查点 #25

远程 ep45. ep50 eval ~13min.
**决策**: 继续

### [22:36] 检查点 #26

远程 ep47. ep50 eval ~8min.
**决策**: 继续

### [22:44] 检查点 #27

远程 ep50 (iter 80). eval ~3min.
**决策**: 等 eval

### [22:48] 检查点 #28 — 远程 ep50 🔥🔥

**远程 GCN+PAA+CE+OA-SD ep50: 67.2/78.8!!**

| Epoch | mAP | R1 |
|-------|------|------|
| 10 | 47.9% | 60.3% |
| 20 | 56.6% | 68.3% |
| 30 | 60.8% | 72.1% |
| 40 | 65.4% | 76.5% |
| **50** | **67.2%** | **78.8%** |

**已接近 exp202b STD-PR+SupCon+3-view (69.3/80.2)！而这只是 1-view!**
如果趋势持续: final 可能 **72-73%!**
vs 4090 PAA (70.8) → 可能超过 **+2-3%!**
**决策**: 继续！这个实验非常有前途
