# exp105 SGRE 审查报告

## 审查范围
- `experiments/exp105/design.md`
- `experiments/exp105/monitor.md`
- `configs/occluded_duke/pose_psg_gcn_paa_sgre.yml`
- `model/modules/skeleton_reencoder.py`
- `model/pose_backbone_model.py`
- `processor/processor.py`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | HIGH | design.md vs repo | 设计要求 `scripts/eval_sgre.py` 在测试时对 top-K 做 SGRE re-rank，但仓库里没有这个脚本，模型 inference 也从未调用 SGRE | 未修复 |
| 2 | HIGH | processor.py | 训练时输入 SGRE 的 `kp_feats` / `kp_weights` 被全部 `detach()`，因此 SGRE loss 只训练 SGRE 模块本身，不会反向塑造 backbone/GCN 特征 | 未修复 |
| 3 | MEDIUM | skeleton_reencoder.py | 代码里构造了 visibility-based `attn_mask`，但调用 `MultiheadAttention` 时并没有真正传入这个 mask，visibility 并未参与 cross-attention 本身 | 未修复 |
| 4 | MEDIUM | exp105 全体 | 只有一条“已启动”的 monitor 记录，没有完成训练或评估证据 | 未修复 |

## 审查通过项

- `SkeletonReEncoder` 已正确注册进模型，参数会被优化器更新
- SGRE similarity head 和 triplet-style training loss 的局部实现是自洽的

## 结论

❌ **不通过**

`exp105` 当前只是一个训练期单独学习的 pair scorer 原型，不是 design.md 里那个真正进入检索闭环的 SGRE 方法。
