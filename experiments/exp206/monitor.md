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
