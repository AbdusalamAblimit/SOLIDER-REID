# 实验 exp136: Corrected Changed-Pair Sparse LPCS

## 动机

`exp134` 原本 intended 要验证：

1. `LPCS` 已经是 pair-specific correction scorer
2. 但其 ranking supervision 仍可能被大量低 teacher-change pairs 稀释
3. 因此有必要只保留 changed-pair 最大的那一部分 pair 参与监督

但 `exp134` 与 `exp133` 一样，由于共享接线 bug 成为失效 run。  
因此 `exp136` 的角色是：

**在修复共享接线 bug 后，第一次真实验证 sparse changed-pair LPCS。**

## 核心假设

如果 `LPCS` 当前的瓶颈真是 supervision dilution，那么：

1. 相对 corrected `exp135`
2. 只保留每个 anchor 的 top teacher-change pairs
3. 应更集中地把梯度打到真正被 support-complete teacher 改变的关系上

## 技术方案

相对 `exp135`，只改一个核心变量：

1. `POSE_LPCS_PAIR_MODE = 'delta_top'`
2. `POSE_LPCS_PAIR_TOP_RATIO = 0.25`

其余全部保持一致：
- scorer
- descriptor
- teacher bank
- `cvk_residual`
- 优化器与训练设置

## 对照组

1. 直接对照：`exp135 Corrected LPCS`
2. 历史参考：`exp134` 仅作为失效 run 记录，不作为方法结果对照

## 预期结果

1. `lpcs_psr` 应显著低于 `1.0`
2. `lpcs_pf` 应高于 `1.0`
3. 若假设成立，`exp136` 应比 `exp135` 更早兑现中后期收益

## 风险与失败解释

1. 如果 `lpcs_psr` 没有降下来：
   - 说明 sparse routing 仍未真实生效
2. 如果 `lpcs_psr` 降下来了但指标变差：
   - 说明 `LPCS` 不能像 `SCRD` 那样过度稀疏
3. 如果 `exp136` 优于 `exp135`：
   - 才能说明 supervision dilution 是 `LPCS` 的真实瓶颈
