# exp004: Pose Feature Modulation (PFM) + Part Pooling

## 实验配置
- **分支**: `exp/pose_heatmap`
- **Config**: `configs/occluded_duke/pose_pfm.yml`
- **输出目录**: `./log/occluded_duke/exp004_pfm`
- **GPU**: RTX 3090 24GB
- **PID**: 591453

## 核心创新
**PFM（Pose Feature Modulation）**：在 backbone 特征图和 pooling 之间插入 pose-conditioned 特征调制。
- 现有方法仅用 pose 做空间选择（WHERE to pool）
- PFM 用 pose 做特征调制（WHAT to enhance/suppress）
- 两者正交，同时改善 global 和 part features

### 技术细节
- PFM: 17ch heatmap → Conv1x1(17→64) → ReLU → Conv1x1(64→768) → modulation weights
- 应用：feat_map × (1 + mod_weights)，零初始化使初始状态为恒等映射
- 参数量：~51K（极轻量）
- Global feature 从调制后的特征图 GAP 得到

## 与前序实验的区别
| | exp001 | exp004 |
|--|--------|--------|
| PFM | ❌ | ✅ |
| Part pooling | sigmoid | sigmoid |
| Global/Part weight | 50/50 | 50/50 |
| Test feat | part-only | part-only |
| Global feat from | 原始特征图 GAP | PFM调制后特征图 GAP |

## Baseline 参考
- exp000 Baseline: mAP 56.6%, R1 66.5%
- exp001 (sigmoid, part-only): mAP 57.5%, R1 67.1%
- exp002 (spatial_softmax, part-only): mAP 57.5%, R1 66.8%

## 监控日志

---
### [15:40] 检查点 #1

**状态**: 🟢正常
**进度**: Epoch 1/120 (0.8%)

| 指标 | exp004 (ep1 iter100) | exp001 同期 |
|------|---------------------|------------|
| Total Loss | 15.868 | 14.323 |
| id_global | 6.554 | 6.554 |
| id_part | 6.554 | 6.554 |
| tri_global | 9.307 | 7.766 |
| tri_part | 9.321 | 7.773 |

**观察**: Loss 模式与 exp001 几乎相同。id_global 和 id_part 初始值一致（6.554/6.554），符合预期——PFM 零初始化使初始状态等价于 exp001。tri_global/tri_part 略高于 exp001，这是正常的初始波动。

**决策**: 继续训练。

