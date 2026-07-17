# 实验 exp125: Sparse Pair-Delta SCRD

## 动机

`exp123` 的正式评估说明：pair-level `teacher-change focusing` 方向本身没有被否定，但 `alpha=1.0` 的连续 `delta` 加权只得到与 `exp119` 近乎等价的最终结果。与此同时，远程 `exp124` 到 `ep40` 已说明单纯把 `alpha` 提高到 `4.0` 只能显著放大 `pair_focus`，还没有把中期指标稳定拉开。

当前更合理的解释是：teacher-change pairs 本来就稀疏，若仍对所有 pair 做连续平滑加权，真正高信息量的 changed pairs 仍会被大量近零变化 pair 稀释。

## 核心假设

如果只让每个 anchor 下 **teacher-change 最大的一小部分 pair** 参与 `CSRD`，而不是对所有 pair 做连续加权，那么 support-complete relational teacher 的有效监督会更集中、更有机会转成正式 eval 增益。

## 技术方案

相对 `exp123`，只改 `CSRD` 的 pair 路由机制：

1. 保持 support-complete teacher、bank 更新、主损失配比完全不变
2. 新增 `POSE_CSRD_PAIR_WEIGHT_MODE = 'delta_top'`
3. 对每个 anchor 的正/负 pair 子集分别按 `pair_delta` 排序
4. 仅保留 top-`25%` 的 teacher-change pairs 参与 `CSRD`
5. 未被选中的 pair 在 distillation logits 中被稀疏屏蔽

额外记录：
- `csrd_psr`: 实际被保留的 pair 比例
- `csrd_pf`: 被保留 pair 的平均 focus

## 对照组

- 直接对照: `exp123 Pair-Delta Focused SCRD`
- 辅助对照: 远程仍在运行的 `exp124 alpha=4.0`

## 预期结果

1. `csrd_psr` 应显著低于 `1.0`，证明稀疏 pair routing 已真实生效
2. 如果假设正确，`exp125` 应在 `ep30/40` 起比 `exp123` 更早兑现增益
3. 若正式 eval 转正，则说明当前主线真正需要的是 **sparse pair routing**，而不是继续扫平滑放大系数

## 风险与失败解释

1. 如果 `csrd_psr` 过低且指标变差，说明路由过稀疏，削弱了有效监督总量
2. 如果 `csrd_psr` 生效但结果仍近乎等价，说明当前瓶颈不在 pair 路由稀疏性，而更可能在 teacher 本身的区分度
3. 如果 `exp124` 后续明显优于 `exp125`，则说明当前问题更偏向“强度不够”，而不是“连续加权过于平滑”
