# exp042: `equal_concat` vs `cvk_hybrid` 的 pair-case 差分分析 — 监控日志

## 实验概述
- **目的**: 解释 `cvk_hybrid` 的 `+0.8% mAP` 到底来自哪些 query-level 排序变化
- **checkpoint**: `log/occluded_duke/exp030a_psg_gcn/transformer_120.pth`
- **比较模式**:
  - `equal_concat`
  - `cvk_hybrid (1:1)`
- **输出目录**: `log/occluded_duke/exp042_pair_case_analysis`

## 运行前检查
- [x] `exp040 / exp041` 已确认当前对照口径
- [x] 不改默认评测逻辑，只新增独立分析脚本
- [ ] 运行差分分析脚本
- [ ] 回填 aggregate summary 与典型样例统计
