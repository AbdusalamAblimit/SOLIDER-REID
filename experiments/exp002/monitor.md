# exp002: Spatial Softmax Part Pooling

## 实验配置
- **分支**: `exp/pose_heatmap`
- **Config**: `configs/occluded_duke/pose_spatial_softmax.yml`
- **输出目录**: `./log/occluded_duke/exp002_spatial_softmax`
- **GPU**: RTX 3090 24GB

## 与 exp001 的唯一区别
- 热图归一化：sigmoid → spatial_softmax (temperature=1.0)
- Spatial softmax 在每个 part 的 12×4=48 个位置上做 softmax，产生更尖锐的注意力

## Baseline 参考（来源: log 文件）
- exp000 Baseline: mAP 56.6%, R1 66.5%
- exp001 Sigmoid: mAP 57.1%, R1 66.7%

## 监控日志

---
### [12:01] 检查点 #1

**状态**: 🟢正常
**进度**: Epoch 1/120 (0.8%)

| 指标 | exp002 (ep1 iter200) | exp001 同期 |
|------|---------------------|------------|
| Total Loss | 13.551 | 14.323 |
| id_global | 6.554 | 6.554 |
| id_part | 6.554 | 6.554 |
| tri_global | 6.974 | 7.766 |
| tri_part | 7.020 | 7.773 |

**观察**: 无 inf。tri_part 比 tri_global 大 0.046（exp001 中仅差 0.007），说明 spatial softmax 确实让 part 特征更独特（距 GAP 更远）。初始损失整体比 exp001 低，可能因为 spatial softmax 的注意力更集中。
**决策**: 继续。关注 id_part 是否比 exp001 更快下降。

---
### [12:10] 检查点 #2 — Epoch 10 评估

**状态**: 🟡对比中
**进度**: Epoch 10/120 (8.3%)

| 指标 | exp002 | exp001 ep10 | Baseline ep10 |
|------|--------|------------|--------------|
| mAP | 30.8% | 31.8% | 33.1% |
| R1 | 40.0% | 41.3% | 42.5% |
| R5 | 55.6% | 56.2% | 59.1% |
| R10 | 62.4% | 62.7% | 65.2% |
| id_part | 6.374 | ~6.37 | N/A |

**观察**: ep10 时 exp002 比 exp001 略差 1%。id_part 下降速度与 exp001 相当（6.374 vs ~6.37）。但 spatial softmax 的 tri_part (0.591) > tri_global (0.574)，差异比 exp001 大。早期阶段差异可能不明显。
**决策**: 继续训练，关键观察点在 ep40-60 — 这是 exp001 开始追赶 baseline 的阶段。

