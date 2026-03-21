# 实验 exp134: Changed-Pair Sparse LPCS

## 动机

`exp133 LPCS` 把 learned pair module 从 `alpha-fusion` 升级成了 `residual correction + ranking-aligned supervision`，但它当前仍对每个 anchor 的所有正负 pair 做连续 teacher-change 加权。

如果真正高信息量的 supervision 只集中在一小部分 teacher-change 较大的 pair 上，那么这种“全 pair 连续加权”仍可能存在明显的监督稀释。

这与 `exp125` 在 `SCRD` 线里暴露出的现象一致：

1. pair focus 方向本身有效
2. 但连续平滑加权不如更结构化的 sparse routing

因此，`exp134` 要验证的不是新的 scorer，也不是新的 teacher，而是：

**LPCS 的 ranking supervision 是否也需要 changed-pair sparse routing。**

## 核心假设

如果 `exp133` 的当前瓶颈是 supervision dilution，那么：

1. 只让 teacher-change 最大的一部分正/负 pair 参与 ranking loss
2. 会比“所有 pair 连续加权”更集中地把梯度打到真正发生 support-complete correction 的 pair 上
3. 从而比 `exp133` 更容易兑现为正式检索增益

## 技术方案

相对 `exp133`，只改 `LPCS` 的 pair 路由机制：

1. 保持 pair descriptor、`PairResidualScorer`、support-complete teacher bank、测试期 `cvk_residual` 完全不变
2. 新增：
   - `POSE_LPCS_PAIR_MODE = 'delta_top'`
   - `POSE_LPCS_PAIR_TOP_RATIO = 0.25`
3. 对每个 anchor：
   - 正样本对内按 `pair_change` 取 top-25%
   - 负样本对内按 `pair_change` 取 top-25%
4. 仅保留这些 selected pairs 进入 ranking loss
5. 额外记录：
   - `lpcs_psr`: 实际保留 pair 比例
   - `lpcs_pf`: 保留 pair 的平均 focus 强度

## 对照组

1. 直接对照：`exp133 LPCS`
2. 间接背景对照：`exp125 Sparse Pair-Delta SCRD`

## 预期结果

如果假设成立：

1. `lpcs_psr` 应显著低于 `1.0`
2. `lpcs_pf` 应明显高于 `1.0`
3. `exp134` 应在 `epoch 30+` 起比 `exp133` 更早或更明显地拉开
4. 如果正式 eval 也转强，则说明：
   - learned pair scorer 不是问题
   - 当前真正缺的是 sparse changed-pair supervision

## 风险与失败解释

1. 如果 `lpcs_psr` 生效但结果变差：
   - 说明 `LPCS` 的 ranking supervision 不能像 `SCRD` 那样过于稀疏
2. 如果 `lpcs_psr` 接近预期而结果仍与 `exp133` 等价：
   - 说明当前瓶颈不在 supervision dilution，而更可能在 pair descriptor 或 scorer 形式
3. 如果 `lpcs_psr` 没有降下来：
   - 说明当前实现没有形成真正稀疏路由，需要先修机制而不是解释结果
