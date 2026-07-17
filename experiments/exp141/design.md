# 实验 exp141: Competition-Context LPCS

## 动机

`exp139 query-context` 说明 pair correction 确实需要语境，但它使用的是 query 级统计摘要：

1. `row_mean / row_std / row_min`
2. `row_support_mean / row_gap_mean`

这类摘要只能告诉模型“这个 query 整体难不难”，却不能告诉模型：

- 当前这个 candidate 在所有候选里到底排第几
- 当前这个 pair 的 common-support 改善是普遍现象还是稀有现象

如果真正影响检索的是 **candidate competition**，那么仅靠 query 级均值摘要可能仍然太粗。

## 核心假设

如果给 `LPCS` scorer 增加 **query 内部的 candidate competition context**，让它看到：

1. 当前 pair 的 `base_dist` 在本 query 全部候选中的相对排名
2. 当前 pair 的 `kp_dist` 相对排名
3. 当前 pair 的 `support_ratio` 相对排名
4. 当前 pair 的 `base_dist - kp_dist` 改善在本 query 中是否属于少数显著 pair

那么模型会比 `query_ctx` 更懂：

- 什么时候 common-support correction 值得强用
- 什么时候只是“大家都差不多”的泛化信号，不该过度修正

## 技术方案

在 `LPCS` 的 pair descriptor 基础 6 维上，新增 5 维 **competition context**：

1. `base_rank`
2. `kp_rank`
3. `support_rank`
4. `gain_rank`，其中 `gain = base_dist - kp_dist`
5. `gain_zscore`

于是输入维度从 6 维变为 11 维，但与 `exp139` 不同，这 5 维不是 query 级常数广播，而是 **pair-specific relative-position features**。

训练与测试都按当前 query 的 candidate set 直接构造：

- 训练：batch 内，排除对角线
- 测试：query 对全部 gallery

## 对照组

1. `exp135 Corrected LPCS`
   - 无额外 context
2. `exp139 Query-Context LPCS`
   - query 级摘要 context

## 预期结果

如果假设成立，应看到：

1. 相对 `exp135`，中后期 `R1` 更容易被抬起
2. 相对 `exp139`，若 query 级均值摘要不够，则 competition-context 会在 `R1` 上更强
3. `lpcs_ctxm` 明显大于 0，证明 competition context 确实接入

## 风险与失败解释

1. 如果结果与 `exp139` 基本等价：
   - 说明真正需要的不是更细的相对排名语境，而是别的机制
2. 如果 `mAP` 涨但 `R1` 不涨：
   - 说明 competition context 仍然偏平滑，不足以抓到 top-rank errors
3. 如果明显变差：
   - 说明 relative ranking features 太 noisy，或者 batch 内 competition 与真实 gallery competition 差异太大
