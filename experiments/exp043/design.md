# 实验 exp043: CVK case study 可视化

## 动机
- `exp042` 已经告诉我们：
  - `cvk_hybrid` 的收益主要来自 deeper-rank correction
  - 但同时也会带来少量 top-1 退化
- 仅有统计还不够，论文素材需要具体样例来展示：
  - 哪些 query 被修好
  - 哪些 query 被破坏
  - `equal_concat` 与 `cvk_hybrid` 的 top-1 差异长什么样

## 核心假设
- 如果 `cvk_hybrid` 确实在修正共同可见支撑不足的 pair，那么在最典型的改进样例中，`cvk_hybrid` 的 top-1 应更符合 query 的可见局部证据。
- 退化样例也应能帮助界定它的边界。

## 技术方案
- 输入：`exp042` 生成的 `query_deltas.csv`
- 输出：
  - `top_improved.png`
  - `top_degraded.png`
- 每行展示：
  - query
  - `equal_concat` 的 top-1
  - `cvk_hybrid` 的 top-1
  - `delta_ap / rank change / status`

## 对照组
- baseline: `equal_concat` 的 top-1 检索结果
- target: `cvk_hybrid` 的 top-1 检索结果

## 预期结果
- 能挑出一组对 story 有代表性的改进样例
- 也能保留退化样例，避免只做单边 cherry-pick

## 风险与失败解释
1. top-1 画面不一定能完整反映 AP 改善，因为有些收益来自 deeper ranks。
2. 若退化样例更显眼，也不等于 aggregate 结论失效；它只是说明 trade-off 需要如实呈现。
