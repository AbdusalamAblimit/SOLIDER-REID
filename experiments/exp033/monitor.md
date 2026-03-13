# exp033: Target Person Assignment — 监控日志

## 分析结果

### 多人图统计

| Split | 总图片数 | 多人图 | 占比 | 2人 | 3人 | 4人 | 5人 | 6人 |
|-------|---------|--------|------|-----|-----|-----|-----|-----|
| train | 15618 | 4127 | 26.4% | 2545 | 1064 | 354 | 116 | 48 |
| query | 2210 | 1090 | 49.3% | — | — | — | — | — |
| gallery | 17661 | 4534 | 25.7% | — | — | — | — | — |

### Targetness 打分系统

四个因子加权组合（总权重 = 1.0）：
- `center_score` (0.35): bbox 中心距离 crop 中心的归一化距离
- `area_score` (0.30): bbox 面积 / crop 面积（归一化）
- `containment` (0.20): bbox 在 crop 内的面积占比
- `mean_score` (0.15): 17 个关键点的平均置信度

### Person 0 vs Targetness 打分的一致性

| Split | 多人图 | Person 0 = Target | Person 0 ≠ Target | 一致率 |
|-------|--------|-------------------|-------------------|--------|
| train | 4127 | 4051 | 76 | **98.2%** |
| query | 1090 | 1052 | 38 | **96.5%** |
| gallery | 4534 | 4467 | 67 | **98.5%** |

### Margin 分布（衡量 target 选择的置信度）

| Split | Mean | Std | <0.02 (极不确定) | <0.05 (不确定) | >0.10 (确定) |
|-------|------|-----|-----------------|---------------|-------------|
| train | 0.347 | 0.165 | 45 (1.1%) | 113 (2.7%) | 3848 (93.2%) |
| query | 0.239 | 0.141 | 28 (2.6%) | 83 (7.6%) | 912 (83.7%) |
| gallery | 0.336 | 0.165 | 50 (1.1%) | 148 (3.3%) | 4191 (92.4%) |

### 视觉审查

生成了 200 张随机多人图可视化 + 76 张全部 disagreement 案例。

**可视化审查结论：**
1. **正常案例（98%+）**: person 0 明显是中心/最大的人物，其他人是边缘的部分可见旁人。target assignment 正确。
2. **Disagreement 案例（~2%）**: 大多数是两人高度重叠、面积接近的场景。这些案例中 targetness 打分选择的更中心的人通常看起来更合理，但由于极度重叠，两个选择都不完美。
3. **典型 disagreement 模式**:
   - 一个稍大但偏离中心的人 vs 一个稍小但更中心的人
   - 两人几乎完全重叠（bbox IoU > 0.7），得分非常接近
   - 极端拥挤场景（5-6 人），所有人都部分可见

**通过标准**: >=85% 明显正确 ✅（实际 >98% 正确，剩余 <2% 为真正模糊的边界情况）

### 结论

1. **现有的"按面积排序"启发式已经非常强**：98.2% 的多人图中 person 0 就是最佳 target
2. **Targetness 打分提供了边界情况的改善**：修正了 1.5-3.5% 的图片，主要是"大但偏心"vs"小但居中"的情况
3. **对训练的预期影响很小**：只有约 2% 的多人图（即约 0.5% 的总训练图）会被重新分配，因此 exp034 的性能变化预计在噪声范围内
4. **但代码健壮性显著提升**：为后续 visibility 实验提供了正确的 target-aware 基础设施

### 产出文件
- `scripts/compute_target_assignment.py` — targetness 计算脚本
- `scripts/visualize_target_assignment.py` — 可视化脚本
- `data/occluded_duke/pose_data/{split}/index.json` — 已更新，包含 target_person_idx, target_score, target_margin, person_targetness
- `experiments/exp033/visualizations/` — 200 张随机多人图可视化
- `experiments/exp033/visualizations/disagreements/` — 76 张 disagreement 可视化
