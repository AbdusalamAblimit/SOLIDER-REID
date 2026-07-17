# exp006: L2-Normalized Feature Fusion — 测试结果

## 实验说明
不需要重新训练。使用 exp001 已训练模型，测试不同 test-time 特征融合策略。

## 结果（使用 exp001 模型, 120 epoch）

| Mode | mAP | R-1 | R-5 | R-10 | 描述 |
|------|-----|-----|-----|------|------|
| global | 57.1% | 66.7% | 78.5% | 83.0% | 仅 global 特征 |
| part (scaled) | 57.5% | 67.1% | 79.1% | 83.5% | 仅 part 特征 (1/N scale) |
| part_noscale | 57.5% | 67.1% | 79.1% | 83.5% | 仅 part 特征 (无 scale) |
| norm_part_only | 57.5% | 67.1% | 79.0% | 83.5% | 仅 part (每个 part L2-norm) |
| **norm_concat** | **57.4%** | **66.9%** | **78.9%** | **83.5%** | L2-norm(global) + L2-norm(parts) |
| equal_concat | 57.4% | 67.0% | 78.9% | 83.5% | 等权拼接 |
| concat (1/N scale) | 57.2% | 66.6% | 78.5% | 83.0% | 原始方式 |

## 分析

1. **Part-only 始终最好** (57.5%)：global feature 的加入不仅没有帮助反而轻微降低
2. **L2-norm 改善了 concat** (57.2→57.4%)：但仍不如 part-only
3. **1/N scaling 有害**: concat (57.2%) < equal_concat (57.4%)，确认 scaling 稀释信号
4. **Part 内部 scaling 无影响**: part_noscale = part (scaled)，因为余弦距离对均匀缩放不变

## 结论

融合方式的优化空间很小（57.2→57.5% 之间）。Part features 与 global features 在距离度量空间中存在轻微冲突。**这个方向的上限已到 +0.9% mAP（part-only），无法通过融合方式进一步突破。**
