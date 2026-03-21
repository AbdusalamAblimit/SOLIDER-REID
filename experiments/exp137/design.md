# 实验 exp137: Hard-Rank LPCS

## 动机

`exp135` 已证明 corrected full-pair `LPCS` 确实有效，但其最终形态是：
- `mAP` 更强
- `R1` 更弱

`exp136` 又证明真稀疏 routing 已经被正确实现，但到 `ep70` 为止并没有自动转成更强指标。  
这说明当前最值得怀疑的瓶颈，不再是 pair routing 语义，而是 `LPCS` 的 **ranking 聚合方式** 过于平均，导致监督更像在做 deeper-rank correction，而不是修 hardest/top-ranked 错误。

## 核心假设

如果把 `LPCS` 从“对所有 selected pos-neg 组合做平均 softplus”改成“只关注每个 anchor 内 hardest 的一小部分 pos / neg”，那么：

1. `LPCS` 会更贴近最终检索的 top-rank 错误
2. `R1` 会比 `exp135` 更有希望转正
3. 如果这条线成立，说明当前主瓶颈确实在 ranking objective，而不是 sparse routing 本身

## 技术方案

基于 `exp135` 保持以下内容全部不变：
- backbone / batch size / 优化器
- online support teacher
- `pair_mode=all`
- evaluator / test-time `cvk_residual`

只改一个核心变量：
- `POSE_LPCS_RANK_MODE: 'hard_top'`
- `POSE_LPCS_RANK_TOP_RATIO: 0.25`

具体机制：
1. 先按 `exp135` 的 full-pair 方式得到 routed positives / negatives
2. 再在每个 anchor 内做 **hard ranking selection**
   - positive 保留距离最大的 top-25%
   - negative 保留距离最小的 top-25%
3. 只对这些 hardest pos-neg 组合计算 `LPCS` ranking loss

额外日志：
- `lpcs_rsr`: rank-level selected ratio  
  用于确认 hard ranking 真的激活，而不是退化成全保留

## 对照组

- 直接对照：`exp135 Corrected LPCS`
- 间接参考：`exp136 Corrected Sparse LPCS`

## 预期结果

理想形态不是单纯继续抬高 `mAP`，而是：
- 相对 `exp135`
  - `R1` 更强
  - `mAP` 至少不明显回撤

如果 `lpcs_rsr` 明确小于 `1.0`，且 `R1` 转正，则支持：
- 当前瓶颈主要在 ranking aggregation

## 风险与失败解释

1. 如果 early stage 明显更差：
   - hard mining 可能过于激进，正负样本太少，噪声过大
2. 如果 `mAP` 掉、`R1` 也不涨：
   - 说明当前 `LPCS` 不是“目标太平均”，而更像 head 表达能力或 supervision source 本身不够
3. 如果 `lpcs_rsr ≈ 1.0`：
   - 说明实现退化，不能解释结果，必须先排查
