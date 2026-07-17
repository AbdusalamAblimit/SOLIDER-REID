# exp043: CVK case study 可视化 — 监控日志

## 实验概述
- **目的**: 生成 `equal_concat` vs `cvk_hybrid` 的改进/退化样例拼图
- **输入**: `log/occluded_duke/exp042_pair_case_analysis/query_deltas.csv`
- **输出目录**: `log/occluded_duke/exp043_case_viz`

## 运行前检查
- [x] `exp042` 已生成 query 级差分 CSV
- [x] 同时保留改进样例与退化样例，不做单边展示
- [x] 运行可视化脚本
- [x] 检查输出图片是否可读

## [11:52] 输出结果

**状态**: ✅ 完成

### 生成文件
- `log/occluded_duke/exp043_case_viz/top_improved.png`
- `log/occluded_duke/exp043_case_viz/top_degraded.png`

### 同步到论文素材目录
- `experiments/paper_materials/figures/qualitative/cvk_top_improved.png`
- `experiments/paper_materials/figures/qualitative/cvk_top_degraded.png`

### 产物检查
- 两张图尺寸均为 `456 x 3710`
- 每张图包含 `8` 行样例
- 每行展示：
  - query
  - `equal_concat` top-1
  - `cvk_hybrid` top-1
  - `status / delta_ap / rank_gain`

### 观察
1. 改进图中同时包含：
   - `top1_fixed`
   - `both_top1_wrong but rank improved`
   - `both_top1_correct but AP improved`
   这和 `exp042` 的统计结论一致，说明收益并不只来自单一种类样例。
2. 退化图被完整保留，避免 qualitative 部分只做单边 cherry-pick。
3. 这组图已经可以直接作为论文 qualitative section 的候选素材。
