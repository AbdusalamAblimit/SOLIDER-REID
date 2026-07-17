# exp211 Small + GCN+PAA+CE+OA-SD + MST (MaxSim Triplet) 监控

配置: exp206r + MaxSim Triplet loss (weight=0.5, margin=0.3)
对照: exp206r (70.6/82.6 eq, 72.3/82.9 maxsim), exp210b (70.6/81.8 eq, 72.4/83.1 maxsim)

## 检查点

### [08:22] 检查点 #1

**状态**: 正常
**进度**: Epoch 1/120

| 指标 | 当前值 |
|------|--------|
| oa_sd | 0.340 |
| mst | 0.508 (MaxSim triplet loss) |
| id_global | 6.553 |

MST loss = 0.508 (正常初始值，接近 margin=0.3)。
**决策**: 继续

### [08:28] 检查点 #2

ep3. mst=0.315 (从 0.508 快速下降——MaxSim triplet 在学习!)。
oa_sd=0.410. ETA 5h40m.
ep10 eval ~22min.
**决策**: 继续

### [08:52] 检查点 #3 — ep10 ✓

**ep10: 50.4/63.9** (= exp206r 50.4/63.9 — 完全一致!)
MST weight=0.5 不干扰 CE 收敛（与 PKC=0.5 的灾难完全不同）。
mst=0.295 (接近 margin=0.3，triplet 在学习)。
**决策**: 继续

### [09:11] 检查点 #4

ep17. mst=0.290 (平台，接近 margin). ep20 eval ~10min.
**决策**: 等 eval

### [09:19] 检查点 #5

ep20 mid. mst=0.290. eval ~4min.
**决策**: 等 eval

### [09:23] 检查点 #6 — ep20 ✓

**ep20: 56.6/68.0** (= exp206r 56.6/68.1 — 一致!)
MST 不影响 equal_concat 性能。关键在 MaxSim final。
**决策**: 继续到 ep120

### 重大发现: MST/PKC 梯度无法到达 backbone！

对比 exp211 和 exp206r ep10 iter200 的训练 loss：
所有子 loss (id_global, id_part, tri_global, tri_part, oa_sd) **完全一致到小数点后三位！**
唯一不同的是总 Loss (多了 MST 贡献) 和 mst loss 本身。

**根本原因**: `pose_backbone_model.py:434` 中 GCN 输入的 feature map 被 `detach()` 了！
MST/PKC 梯度只能更新 GCN 的 ~200K 参数，无法影响 backbone 的 50M 参数。
而且 MST loss ≈ 0.29 接近 margin=0.3，梯度几乎为零。

**解决方案**: 去掉 detach() 让 per-keypoint loss 的梯度也能更新 backbone。
但这会改变 Part 分支的梯度流，可能影响 baseline 性能。需要谨慎。
