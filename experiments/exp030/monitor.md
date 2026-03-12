# exp030 监控日志: PDS+StopGrad + Skeleton GCN Part Branch

**配置**: `configs/occluded_duke/pose_pds_sg_gcn.yml`
**输出**: `./log/occluded_duke/exp030_pds_sg_gcn`
**对照**: exp023 (PDS+StopGrad, Part Pooling) — mAP 59.5%, R1 68.5%
**核心变量**: Part Pooling → Skeleton GCN
**启动时间**: 2026-03-12 03:48
**PID**: 1295356

---
### [03:49:00] 检查点 #1

**状态**: 🟢正常
**进度**: Epoch 1/120 (0.8%)

| 指标 | 当前值 | 备注 |
|------|--------|------|
| Total Loss | 20.3 | E1 iter40, 正常下降 |
| id_global | 6.555 | |
| id_part | 6.676 | GCN 分支 |
| tri_global | 12.9 | |
| tri_part | 14.4 | GCN 分支 triplet |
| Acc | 0.002 | 刚开始，正常 |
| LR | 4.76e-05 | warmup 阶段 |

**观察**: 模型正确初始化 "[PDS] Part branch: Independent Stage 3 copy + Skeleton GCN (2 layers, hidden=256)"。4 个 loss 分量正常输出。GCN part loss 略高于 global loss（6.676 vs 6.554），符合预期（GCN 从零开始学习）。
**决策**: 继续

---
### [03:51:10] 检查点 #2

**状态**: 🟢正常
**进度**: Epoch 3/120 (2.5%)

| 指标 | 当前值 | 变化趋势 |
|------|--------|----------|
| Total Loss | 8.41 | ↓ 快速下降 |
| id_global | 6.530 | ↓ 缓慢下降 |
| id_part | 5.831 | ↓ 下降中 |
| tri_global | 1.866 | ↓↓ 大幅下降 |
| tri_part | 2.594 | ↓↓ 大幅下降 |
| Acc | 3.5% | ↑ 从 0 开始上升 |
| LR | 1.27e-04 | warmup |
| GPU Mem | 8280 MiB | — |
| GPU Util | 74% | — |

**观察**: Triplet loss 下降非常快（14.4→2.6），说明 GCN skeleton_feat 正在学习区分性特征。id_part 也在下降但慢于 tri_part。~56s/epoch，ETA ~1h49m。
**决策**: 继续

---
### [03:54:15] 检查点 #3

**状态**: 🟢正常
**进度**: Epoch 6/120 (5%)

| 指标 | 当前值 | 变化趋势 |
|------|--------|----------|
| Total Loss | 6.37 | ↓ 持续下降 |
| id_global | 6.321 | ↓ 缓慢 |
| id_part | 4.356 | ↓↓ 从 6.676 降到 4.356 |
| tri_global | 0.796 | ↓ 趋于稳定 |
| tri_part | 1.276 | ↓ 持续下降 |
| Acc | 21.5% | ↑↑ 从 0.2% → 21.5% |

**观察**: id_part 从 6.68 降到 4.36，说明 Skeleton GCN 的分类器正在快速学习。tri_part (1.276) 比 tri_global (0.796) 高，GCN 特征区分度稍弱于 Global 但在正常范围内。Accuracy 爬升正常。
**决策**: 继续
