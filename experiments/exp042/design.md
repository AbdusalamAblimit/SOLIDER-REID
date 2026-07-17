# 实验 exp042: `equal_concat` vs `cvk_hybrid` 的 pair-case 差分分析

## 动机
- `exp040 / exp041` 已确认：
  - `cvk_hybrid` 在 `exp030a` checkpoint 上稳定给出 `+0.8% mAP`
  - `1:1` 是当前测试点中的 mAP 最优权重
- 但目前证据仍停留在 aggregate metric。要把 story 讲清楚，还需要回答：
  - 它到底修正了哪些 query？
  - 改善主要来自 top-1 修复，还是更深层排序改进？
  - 退化样例长什么样？

## 核心假设
- 如果 `cvk_hybrid` 真的是 common-support correction，那么它应主要改善那些：
  - `equal_concat` 下 top-1 错误但存在共同可见局部支撑的 query
  - 或者 AP 有明显提升、但不一定改变 top-1 的 query
- 如果改善分布完全随机，则这条 story 仍然偏弱。

## 技术方案
- 固定 checkpoint：`log/occluded_duke/exp030a_psg_gcn/transformer_120.pth`
- 固定数据与评测口径
- 仅比较两种模式：
  - `equal_concat`
  - `cvk_hybrid (1:1)`
- 新增独立分析脚本，导出：
  - 两种模式的 `distmat`
  - 每个 query 的 AP / first-correct-rank / top1 是否正确
  - 改进 / 退化 query 列表
  - 简要 markdown 总结

## 对照组
- baseline: `equal_concat`
- target: `cvk_hybrid (1:1)`

## 预期结果
- 理想情况：
  - 有一批 query 在 `cvk_hybrid` 下明显提升 AP 或修复 top-1
  - 退化 query 数量更少，且退化幅度更小
- 中性情况：
  - 改进与退化混杂，但改进幅度更大，解释 mAP 上升
- 负情况：
  - 变化分布无明显模式，无法支撑 mechanism-level 解释

## 风险与失败解释
1. `mAP +0.8%` 可能来自大量微小排序变化，而不是少量显著修复。
2. 若 top-1 改善样例很少，也不代表方法无效，可能说明收益主要来自 deeper ranks。
3. 该实验是解释性分析，不应包装成新的性能实验。
