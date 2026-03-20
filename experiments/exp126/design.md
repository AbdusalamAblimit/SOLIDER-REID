# 实验 exp126: Exact Top-K Pair SCRD

## 动机

- `exp123/124/125` 已共同说明：`pair routing` 方向不是负方向，结构化 focus 有效
- 但 `exp125` 也把当前实现的核心缺口暴露得很清楚：
  - `POSE_CSRD_PAIR_WEIGHT_MODE = delta_top`
  - 预期是 sparse routing
  - 实际 `csrd_psr` 长期仍在 `0.89~0.91`
- 这意味着我们到现在并没有真正测试到“强稀疏 pair 选择”，而是在测试一种被 tie 扩散后的温和结构化 focus

因此当前最合理的下一跳不是继续扫 `alpha` 或继续延长训练，而是：
**把 `delta_top` 从阈值近似 top-k，改成 exact top-k pair selection。**

## 核心假设

1. `exp125` 的 `psr` 过高，不是因为 `top_ratio=0.25` 本身无效，而是因为阈值式选法把大量 tie pair 一起保留了
2. 若强制改成 exact top-k，`csrd_psr` 应显著下降到接近 `0.25`
3. 真正更稀疏的 pair routing 会比 `exp125` 更清楚地放大 teacher-change 的有效 pair，从而把当前弱正向进一步兑现

## 技术方案

相对 `exp125`，只改一个核心变量：

- `POSE_CSRD_PAIR_WEIGHT_MODE: delta_top -> delta_top_exact`

实现差异：
- `delta_top`: 用 `delta >= topk_threshold` 的阈值式保留
- `delta_top_exact`: 直接用 `torch.topk` 的 index 构造 exact top-k mask，不允许 tie 扩散

其余保持完全不变：
- `POSE_CSRD_PAIR_TOP_RATIO = 0.25`
- `POSE_CSRD_PAIR_WEIGHT_ALPHA = 1.0`
- support-complete teacher 构造不变
- bank 更新规则不变
- backbone、batch size、主 loss 配比不变

## 对照组

- 直接对照：`exp125 Sparse Pair-Delta SCRD`
- 强度对照：`exp124 Stronger Pair-Delta SCRD`
- 上一层方法对照：`exp123 Pair-Delta Focused SCRD`
- 主基线：`exp030a`

## 预期结果

- 若假设成立：
  1. `csrd_psr` 应从 `0.90` 左右降到接近 `0.25`
  2. `pair routing` 的收益形态应比 `exp125` 更清楚
  3. `ep30/40/50` 至少有一段应稳定优于 `exp125`

- 若失败：
  1. 说明当前瓶颈不只是“稀疏性没打到位”
  2. 更可能需要改的是 pair selection 依据本身，而不是 top-k 实现细节

## 风险与失败解释

1. 若 exact top-k 过硬，可能把一部分中等价值 pair 一并丢掉，导致 `R1` 波动
2. 若 `delta` 分布本身噪声较大，强稀疏选择可能会放大错误 teacher-change pair
3. 若结果不如 `exp125`，说明当前最好的点可能已经落在“温和结构化 focus”，下一步应转向改 selection signal，而不是继续加硬 mask
