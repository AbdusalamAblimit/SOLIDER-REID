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
- [x] 运行差分分析脚本
- [x] 回填 aggregate summary 与典型样例统计

## [11:48] 分析结果

**状态**: ✅ 完成

### Aggregate
- `equal_concat` = `61.1% mAP / 73.7% R1`
- `cvk_hybrid` = `61.9% mAP / 73.2% R1`
- delta = `+0.7% mAP / -0.5% R1`

### Query 级统计
- query 总数: `2210`
- `positive_delta_ap`: `1129`
- `negative_delta_ap`: `822`
- `zero_delta_ap`: `259`
- `mean_delta_ap`: `+0.00737`
- `median_delta_ap`: `+0.00020`
- `top1_fixed`: `47`
- `top1_degraded`: `58`
- `both_top1_correct`: `1571`
- `both_top1_wrong`: `534`

### 观察
1. mAP 增益来自 **更广泛的 AP 重分配**，而不是只靠少数 query 的偶然修复：
   正向 query (`1129`) 明显多于负向 query (`822`)。
2. `top1_fixed (47)` 少于 `top1_degraded (58)`，这与 `R1 -0.5%` 完全一致。
3. 因而 `cvk_hybrid` 的主要作用不是“让更多 query 的 top1 立刻翻正”，而是 **改善大量 query 的整体排序质量**。
4. 最强改进样例里既有：
   - `rank 2 -> 1`
   - `rank 4 -> 1`
   - `rank 9 -> 1`
   也有：
   - `rank 84 -> 3`
   - `rank 11 -> 2`
   这说明它确实会修正一部分困难遮挡 pair，而不只是对 easy case 轻微扰动。
5. 同时也存在 `top1_degraded` 样例，说明当前 common-support reasoning 还不是无代价增强。

## exp042 阶段结论
1. `cvk_hybrid` 的收益形状现在可以被解释为：
   **用少量 R1 代价，换取更多 query 的 AP 改善。**
2. 这和前面的 aggregate metric 完全一致：
   - `mAP +0.8`
   - `R1 -0.5`
3. 因此目前最稳的 story 不是“CVK 更适合冲 top1”，而是：
   **CVK 更适合作为 pair-specific deeper-rank correction。**

## 产物
- `log/occluded_duke/exp042_pair_case_analysis/summary.json`
- `log/occluded_duke/exp042_pair_case_analysis/summary.md`
- `log/occluded_duke/exp042_pair_case_analysis/query_deltas.csv`
