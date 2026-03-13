# exp035: Visibility 最小闭环消融 — 监控日志

## 实验概述
- **目的**: 对比 4 种 keypoint pooling 权重模式
- **Base**: exp030a (PSG + GCN, equal_concat)
- **对照**: exp030a seed=1234 equal_concat = 61.1% mAP / 72.9% R1（单 seed 对照）
- **对照 (3-seed)**: exp030a 3-seed mean = 60.73±0.47% mAP / 72.57±0.58% R1

## 重要说明
**exp035a 是一个 bundled sanity check**，不是纯 exp034 验证。它同时包含：
1. exp034 的 target-aware dataloader reordering
2. exp035 的 visibility augmentation fix（flip L-R swap, OOB/erase 清零）
3. 新增的 `_compute_kp_weights()` 代码路径（score mode 下行为等价）

如果 exp035a 与 exp030a seed=1234 差距明显，需补跑纯 exp034 隔离验证。

## 子实验列表
| ID | 权重模式 | Config | 状态 | 最终 mAP | 最终 R1 |
|----|---------|--------|------|----------|---------|
| 035a | score (baseline) | exp035a_kpw_score.yml | ✅ 完成 | 61.1% | 73.8% |
| 035b | score * visibility | exp035b_kpw_score_visibility.yml | ✅ 完成 | 60.4% | 71.6% |
| 035c | visibility only | exp035c_kpw_visibility.yml | 待训练 | — | — |
| 035d | binary visibility | exp035d_kpw_binary_visibility.yml | 待训练 | — | — |

## exp035a: score weighting (bundled sanity check)

### [02:04] 检查点 #1
**状态**: 🟢正常
**进度**: Epoch 1/120

| 指标 | 当前值 |
|------|--------|
| Total Loss | 19.99 (iter 40) |
| id_global | 6.555 |
| id_part | 6.674 |
| tri_global | 12.546 |
| tri_part | 14.196 |
| Acc | 0.001 |
| LR | 4.76e-05 |

**观察**: 训练正常启动，确认 `kp_weight=score`。
**决策**: 继续

### [03:07] 检查点 #2
**状态**: 🟢正常
**进度**: Epoch 60/120 (50%)

| Epoch | mAP | R1 | R10 |
|-------|-----|----|-----|
| 10 | 38.3% | 51.3% | 73.3% |
| 20 | 47.1% | 59.7% | 79.7% |
| 30 | 52.5% | 65.4% | 84.1% |
| 40 | 56.1% | 68.1% | 85.5% |
| 50 | 56.1% | 68.5% | 85.1% |
| 60 | 58.3% | 70.8% | 86.5% |

**观察**: mAP 持续上升，中期趋势健康。
**决策**: 继续

### [04:12] 最终结果
**状态**: ✅ 完成

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 70 | 58.7% | 71.2% | 83.9% | 87.4% |
| 80 | 59.8% | 71.6% | 83.8% | 87.4% |
| 90 | 60.8% | 73.0% | 84.7% | 88.6% |
| 100 | 60.7% | 73.3% | 84.5% | 88.0% |
| 110 | 61.1% | 73.7% | 85.1% | 88.2% |
| **120** | **61.1%** | **73.8%** | **85.1%** | **87.9%** |

**对比 exp030a seed=1234**: mAP 61.1% vs 61.1% (±0.0%), R1 73.8% vs 72.9% (+0.9%)

**结论**: bundled changes 在 score mode 下无 regression。mAP 完全一致，R1 略高但在种子方差范围内。
无需补跑纯 exp034 隔离验证。

## exp035b: score * visibility weighting

### [04:13] 检查点 #1
**状态**: 🟢正常
**进度**: Epoch 1/120

**观察**: 训练正常启动，确认 `kp_weight=score_visibility`。
**决策**: 继续

### [06:20] 最终结果
**状态**: ✅ 完成

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 38.1% | 51.6% | — | 72.7% |
| 20 | 46.2% | 59.0% | — | 79.0% |
| 30 | 51.9% | 64.4% | 78.2% | — |
| 40 | 55.6% | 67.9% | 80.8% | 84.9% |
| 50 | 55.4% | 67.6% | 80.2% | 84.8% |
| 60 | 57.6% | 69.5% | 82.3% | — |
| 70 | 58.0% | 69.7% | 84.0% | — |
| 80 | 59.2% | 70.6% | 83.3% | 87.3% |
| 90 | 60.2% | 72.4% | 84.5% | 88.1% |
| 100 | 60.0% | 71.8% | 84.5% | 88.1% |
| 110 | 60.3% | 71.7% | 84.8% | 88.2% |
| **120** | **60.4%** | **71.6%** | **84.8%** | **87.9%** |

**对比 exp035a (score)**: mAP -0.7% (60.4% vs 61.1%), R1 -2.2% (71.6% vs 73.8%)

**结论**: score_visibility 模式未能提升，反而略差。visibility 的降权效果在此框架下为负。
差距 -0.7% mAP 在种子方差范围内 (3-seed std 0.47%)，但 R1 差距 -2.2% 超过种子方差范围，可能有真实负面影响。

## 总结与决策

### 初步结论
| 模式 | mAP | R1 | vs score (mAP) | vs score (R1) |
|------|-----|----|----------------|---------------|
| 035a: score | 61.1% | 73.8% | — | — |
| 035b: score*vis | 60.4% | 71.6% | -0.7% | -2.2% |
| 035c: vis only | — | — | 待跑 | 待跑 |
| 035d: binary vis | — | — | 待跑 | 待跑 |

### 决策
鉴于 score_visibility（预期最强的 visibility 模式）已表现为负，visibility_only 和 binary_visibility 预计不会更好。
**跳过 035c 和 035d，不再继续 visibility 方向的 keypoint pooling 实验。**
visibility 在 keypoint 级加权池化中无独立价值，scores 已经足够。
