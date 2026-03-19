# exp089 PAMN 审查报告

## 审查范围
- `experiments/exp089/design.md`
- `configs/occluded_duke/pose_psg_gcn_paa_pamn.yml`
- `model/modules/pose_matching_network.py`
- `processor/processor.py`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | HIGH | processor.py | PAMN 不是模型的一部分，而是在 processor 里临时实例化。checkpoint 不会保存它，测试阶段也无法从权重中恢复这个 matching module | 未修复 |
| 2 | HIGH | repo-wide | 设计要求“粗排 top-100 后用 PAMN re-rank”，但仓库里没有 `eval_pamn.py` 或任何测试期接线，PAMN 从未进入实际检索流程 | 未修复 |
| 3 | HIGH | processor.py | 训练时输入 PAMN 的 `kp_feats` / `kp_weights` 全部被 `detach()`，因此 PAMN loss 不会反向塑造 backbone/GCN 表征；如果测试也不用 PAMN，那么主检索路径完全不变 | 未修复 |
| 4 | MEDIUM | exp089 全体 | 没有训练日志或完成的实验记录，当前只有代码草案 | 未修复 |

## 审查通过项

- `PoseMatchingNetwork` 自身的 pair-feature 组装和 triplet-style training loss 在局部上是自洽的

## 结论

❌ **不通过**

`exp089` 现在只是“训练期单独学一个 matcher”，不是 design.md 里那个真正参与检索的 learned matching framework。
