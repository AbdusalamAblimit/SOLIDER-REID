# Codex Review — exp347

**Verdict**: approve
**Date**: 2026-06-20

## 结论
codex 审查通过。PoseWeightedPool 真无参数(无 nn.Parameter/Linear,make_optimizer 不加任何参数);i2t/t2i 对齐梯度经无参数池化直接流进 backbone(featmaps[-1]),这正是 A(exp343=57.6 有 learnable query/k_proj 吸收)的针对性修复;对齐目标是纯 ID 原型(pose_cond False,无 B 的姿态稀释);描述子 = raw GAP global(池化仅 train,无泄漏);scene_heatmaps None fallback;POSE_TEST_FEAT global 与 exp341 描述子等价(单变量成立)。Verdict: approve。

## 变体
exp351 = un-detach + de-occluded, 代码已审. Verdict: approve.
