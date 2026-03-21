# 实验 exp132: LTCS（Learn-to-Trust Common Support）

## 动机

`exp125` 已经证明：**pair-specific correction 是真实有价值的**。  
但接下来的两轮关键实验又把当前主矛盾收紧了：

1. `exp130` 说明：`target form` 不是主瓶颈
2. `exp131` 说明：`relation coverage` 也不是主瓶颈

这意味着当前更合理的解释不再是：
- teacher 不够对
- changed pairs 不够多

而是：

**pair-specific support-complete correction 不适合继续被压进单个 embedding。**

与此同时，固定 `cvk_hybrid` 早就给过稳定正信号，但它始终是手工 `1:1` 融合；  
仓库里虽然有 `exp089 PAMN` 草案，但它从未真正接入 checkpoint 与测试检索流程，因此 learned pair module 这条线实际上还没有被认真做过。

## 核心假设

如果当前瓶颈在于 “correction 的表示形式”，那么最合理的下一步不是再蒸一次 embedding，而是：

**显式学习一个 pair-adaptive fusion rule，决定每个 pair 应该多大程度相信 `global distance` 与 `common-support distance`。**

如果这个假设成立，那么：

1. learned pair fusion 应该优于固定 `cvk_hybrid`
2. 它也应能解释为什么 `exp125` 只得到弱正向：
   - pair correction 值得学
   - 但继续把它压进单向量 global 有上限

## 技术方案

### 1. LTCS Head

在模型中注册一个真正会被保存/加载的 `PairAdaptiveFusionHead`：

- 输入是每个 query-gallery pair 的轻量描述子：
  1. `d_global`
  2. `d_cvk`
  3. `|d_global - d_cvk|`
  4. `common_support_ratio`
  5. `q_visibility_mean`
  6. `g_visibility_mean`
- 输出：
  - `alpha_ij in [0, 1]`

### 2. Pair-Adaptive Distance

测试期不再固定：

`d = 0.5 * d_global + 0.5 * d_cvk`

而改为：

`d_ij = (1 - alpha_ij) * d_global + alpha_ij * d_cvk`

### 3. Support-Complete Teacher Supervision

训练期用 `SupportCompleteBank` 构建更完整的 teacher keypoint distance `d_sc`，监督 `LTCS` 头学会更好的 pair-adaptive fusion：

`L_ltcs = SmoothL1(d_mix, d_sc)`

其中：
- `d_mix` 是 head 预测后的融合距离
- `d_sc` 是 support-complete teacher distance

第一版先让 `LTCS` 主要学习 **fusion rule 本身**，不强行让该 loss 反向塑造 backbone 主干。

### 4. 真正接入检索流程

这一轮必须避免 `exp089` 的老问题：

1. head 必须是模型参数的一部分
2. checkpoint 必须保存它
3. evaluator 必须在测试时真正调用它
4. 结果必须是完整检索结果，不是“训练了一个 matcher，但测试没用”

## 对照组

1. 主基线: `exp030a-eq seed1234`
2. 检索基线: 固定 `cvk_hybrid`
3. 当前最强训练端对照: `exp125`

## 预期结果

如果假设成立：

1. `cvk_adaptive` 应优于固定 `cvk_hybrid`
2. `ltcs_alpha_mean` 不应塌成常数 `0.5`
3. `ltcs_loss` 应稳定下降到合理区间
4. 日志中应看到：
   - 不同 epoch / batch 的 `alpha` 分布确有变化
   - `d_mix` 比固定融合更接近 `support-complete teacher`

## 风险与失败解释

1. `alpha` 塌成常数：
   - 说明 learned rule 实际退化成 fixed `cvk_hybrid`
2. 相对固定 `cvk_hybrid` 没提升：
   - 说明简单标量 fusion rule 仍不够表达 pair-specific correction
   - 下一步应考虑真正的 learned pair scorer，而不是只学 `alpha`
3. 相对 `exp125` 仍无优势：
   - 说明“进入检索路径”本身还不够，后续需要更强的 pair representation
