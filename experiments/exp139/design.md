# 实验 exp139: Query-Context LPCS

## 动机

`exp135` 的 full-pair `LPCS` 能提升 `mAP`，但 `R1` 偏弱。  
这说明当前问题可能不只是“哪些 pairs 该被强调”，还可能是：

**同一个 pair descriptor 在不同 query 上需要不同的 correction 语境。**

也就是说，当前的 `PairResidualScorer` 只看 pair 本身，可能过于短视，缺少：

- 这个 query 整体有多难
- 这个 query 当前的距离分布有多尖锐
- 这个 query 当前的 support 完整度如何
- 这个 query 的 global / common-support 分歧有多大

## 核心假设

如果给 `LPCS` 增加 **无标签且 train/test 一致** 的 query-level context，让每个 pair correction 同时感知：

1. 当前 query 的平均基础距离
2. 当前 query 的距离分布标准差
3. 当前 query 的最小基础距离
4. 当前 query 的平均 common support
5. 当前 query 的平均 global / common-support 分歧

那么：

1. correction 会更稳，不容易过度修正
2. `R1` 有机会比 `exp135` 更好
3. 这能验证“当前瓶颈在 scorer 的上下文建模，而不只是损失聚合”

## 技术方案

相对 `exp135`，只改一个核心变量：

- `POSE_LPCS_CONTEXT_MODE: 'query_ctx'`

具体机制：

1. 保留 `pair_mode=all`
2. 保留原始 6 维 pair descriptor：
   - `global_dist`
   - `kp_dist`
   - `|global_dist - kp_dist|`
   - `support_ratio`
   - `q_vis_mean`
   - `g_vis_mean`
3. 再为每个 query 追加 5 维 query-level context：
   - `row_mean`
   - `row_std`
   - `row_min`
   - `row_support_mean`
   - `row_gap_mean`
4. 最终由 11 维 descriptor 驱动同一个 `PairResidualScorer`
5. 这些 context 特征在训练和测试中使用完全同一套构造逻辑，不依赖 label，也不依赖 oracle teacher

新增日志：

- `lpcs_ctxm`: context mean  
  用于确认 query context 特征确实非零并进入训练

## 对照组

- 直接对照：`exp135 Corrected LPCS`
- 间接参考：`exp138 Rank-Decayed LPCS`

## 预期结果

理想形态：

1. 相对 `exp135`
   - `R1` 更强
   - `mAP` 不明显退
2. 如果成立，说明当前最有效的升级方向不是“更硬的 rank 强调”，而是“更有上下文的 pair correction”

## 风险与失败解释

1. 如果和 `exp135` 完全等价：
   - 说明 query-level context 不足以改变 correction 质量
2. 如果明显变差：
   - 说明这些无标签统计太粗，给 scorer 引入了噪声
3. 如果 `lpcs_ctxm ≈ 0`：
   - 说明 context 特征没有正确接入
