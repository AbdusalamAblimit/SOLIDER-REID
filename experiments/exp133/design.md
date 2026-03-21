# 实验 exp133: LPCS（Learned Pair Correction Scorer）

## 动机

`exp132 LTCS` 已较干净地证明：

1. 检索期 learned pair module 这条大方向没有死
2. 但第一版 `alpha-fusion` 太弱：
   - 只学一个标量 `alpha`
   - 只在 `global distance` 与 `CVK distance` 两个标量之间做凸组合
   - 正式结果与固定 `cvk_hybrid` 完全一致

因此下一步不该继续调 `alpha` 头，而应升级为：

**直接学习 pair-specific correction score / residual score。**

## 核心假设

如果当前真正缺的是 “这个 pair 到底该被修正多少”，那么：

1. 只学 `alpha` 不够表达 pair-specific correction
2. 直接预测 `residual score` 会比固定 convex fusion 更有表达力
3. 如果再用更 ranking-aligned 的监督，learned pair module 才有机会真正超过固定 `cvk_hybrid`

## 技术方案

### 1. Pair Descriptor

先沿用 `exp132` 已验证可落地的轻量 pair descriptor 主体：

1. `d_global`
2. `d_cvk`
3. `|d_global - d_cvk|`
4. `common_support_ratio`
5. `q_visibility_mean`
6. `g_visibility_mean`

必要时再追加更细的 common-support 统计，但第一版尽量保持 descriptor 简洁。

### 2. Residual Pair Scorer

不再输出 `alpha`，而是输出一个标量 residual：

`delta_ij = f(pair_descriptor_ij)`

最终分数/距离为：

`d_final = d_cvk_hybrid + delta_ij`

这样模型学的不是 “该更信谁”，而是：

**这个 pair 相对固定融合应再被修正多少。**

### 3. Ranking-Aligned Supervision

第一版优先考虑：

1. pair label supervision（正负 pair）
2. 或 pairwise margin / ranking loss
3. support-complete teacher 只用于：
   - pair weighting
   - hard pair mining
   - 或 teacher-induced order 监督

核心思想是：不要再只做距离回归，而要更直接地优化排序。

## 对照组

1. `exp132b` 固定 `cvk_hybrid`
2. `exp132a` learned `cvk_adaptive`
3. `exp125` 当前最强的 embedding-side pair correction 版本

## 预期结果

如果假设成立：

1. `LPCS` 应优于同 checkpoint 下固定 `cvk_hybrid`
2. 不应再出现 `exp132` 那种 learned head 与 fixed fusion 完全等价
3. 正式 eval 中至少应看到：
   - 排序结果发生真实变化
   - `mAP` 与 `R1` 不再是纯 trade-off，而出现更明确的净增益

## 风险与失败解释

1. 如果仍然与 `cvk_hybrid` 几乎一致：
   - 说明问题不只是 `alpha` 太弱
   - 可能 pair descriptor 本身也不够表达
2. 如果 `mAP` 涨但 `R1` 持续掉：
   - 说明 scorer 学到的是长尾重排，而不是稳定 top-1 correction
   - 需要进一步约束 ranking objective
3. 如果训练不稳定：
   - 说明 pair scorer 的监督需要更严格地限制在 teacher-changed / hard pairs 上
