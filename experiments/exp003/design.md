# exp003: Part-Dominant Training + Part-Only Testing

## 动机

exp001/002 的特征模式消融发现：
- part-only: mAP 57.5% (+0.9% vs baseline)
- concat (1/N scaled): mAP 57.2% (+0.6%)
- equal concat: mAP 57.4% (+0.8%)
- global only: mAP 57.1% (+0.5%)

**核心发现**: Part 特征单独使用效果最好，加入 global 反而稀释。同时 id_part 收敛极慢（最终仍是 id_global 的 10 倍），说明 0.5/0.5 的 loss 权重对 5 个 part classifier 来说不够。

## 实验假设

增加 part loss 权重（从 0.5 到 0.67），让 backbone 和 part classifiers 获得更强的学习信号。同时测试时只用 part 特征（移除 global concat），消除融合导致的信号稀释。

## 与 exp001 的区别

| 配置 | exp001 | exp003 |
|------|--------|--------|
| Part loss weight | 0.5 (50%) | 0.67 (67%) |
| Global loss weight | 0.5 (50%) | 0.33 (33%) |
| 测试特征 | concat (global + parts/5) | part-only |
| 热图归一化 | sigmoid | sigmoid |

## 技术实现

1. **make_loss.py**: 使用 `POSE_PART_WEIGHT` 控制 part/global loss 比例
   - `w_p = POSE_PART_WEIGHT / (1 + POSE_PART_WEIGHT)`
   - `w_g = 1 / (1 + POSE_PART_WEIGHT)`
   - POSE_PART_WEIGHT=1.0 → 50/50 (default, backward compatible)
   - POSE_PART_WEIGHT=2.0 → 67/33

2. **pose_model.py**: 添加 `POSE_TEST_FEAT` 配置
   - 'concat_scaled' (default, current behavior)
   - 'part_only' (exp003)

## 预期结果

- 如果 part 特征质量提升（id_part 收敛更快），part-only mAP 应该超过 57.5%
- 风险：global loss 减弱可能影响 backbone 学习，但 part loss 仍通过 backbone 反传

## 论文意义

这是消融实验：证明 "part training weight" 和 "test-time fusion strategy" 的重要性。
