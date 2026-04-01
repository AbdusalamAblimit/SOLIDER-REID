# exp208 审查报告

## 审查范围
- design.md 合理性与单变量原则
- config/defaults.py — GLOBAL_LOSS_SCALE 定义
- loss/make_loss.py — GLOBAL_LOSS_SCALE 使用路径
- 与 exp206 的对照关系

## Critical: GLOBAL_LOSS_SCALE 在 GCN+PAA 路线下无效（NO-OP）

**这是一个阻断性问题，不修复则实验无意义。**

`GLOBAL_LOSS_SCALE` 仅在 `make_loss.py` 的 **标量 score/feat 路径** 中使用（L213-214, L253-254）：

```python
# 标量 score path (L213-214):
global_loss_scale = getattr(cfg.MODEL, 'GLOBAL_LOSS_SCALE', 1.0)
ID_LOSS = global_loss_scale * ce_fn(score, target)

# 标量 feat path (L253-254):
global_loss_scale = getattr(cfg.MODEL, 'GLOBAL_LOSS_SCALE', 1.0)
TRI_LOSS = global_loss_scale * triplet(feat, target, ...)[0]
```

但 GCN+PAA 模型返回 **列表** score 和 feat（`pose_backbone_model.py` L517: `return [cls_score] + gcn_cls_scores, [global_feat] + gcn_feats, ...`），所以 loss 函数走 `isinstance(score, list)` 路径（L128-211），该路径使用 `POSE_PART_WEIGHT` 的 `w_g/w_p` 机制，**完全不读取 GLOBAL_LOSS_SCALE**。

换言之，设 `GLOBAL_LOSS_SCALE=0.5` 在 GCN+PAA 路线下会被 **静默忽略**，exp208 训练结果将与 exp206 完全一致（除随机种子差异）。

### 进一步：list-loss 路径已内含隐式 0.5x
当 `POSE_PART_WEIGHT=1.0`（默认值）时，`w_g = 1/(1+1) = 0.5`，`w_p = 0.5`。这意味着 list-loss 路径已经隐式将 global loss 缩放到 0.5x。这正是 exp007a design.md 中记录的："list-loss 已隐式实现 0.5x"（多个 config 文件的注释也确认了这一点）。

## 修复建议

要真正实现"在 exp206 基础上降低 global loss 权重"这一目标，有两种方案：

1. **改 POSE_PART_WEIGHT**：从 1.0 改到更高值（如 2.0），使 w_g = 1/3, w_p = 2/3。这会同时影响 CE 和 triplet 的 global/part 比例。
2. **在 list-loss 路径中引入 GLOBAL_LOSS_SCALE**：修改 `make_loss.py` 的 L132-133 处，将 `w_g` 乘以 `GLOBAL_LOSS_SCALE`。这需要代码修改，但更精确。

但更根本的问题是：**list-loss 路径已经是 0.5x global，与 Tiny 上 exp007a 的 GLOBAL_LOSS_SCALE=0.5 效果一致**。所以"在 Small 上测试 0.5x"这件事实际上 **已经在 exp206 中默认完成了**。如果目标是进一步降低 global（比如 0.25x），则需要用上述方案。

## design.md 审查

- 动机和假设表述清晰，但基于错误前提（认为 exp206 使用 1.0x global）
- 实际上 exp206 的 GCN list-loss 已隐式 0.5x global，所以"从未在 Small 上测试 0.5x"的说法不成立

## 结论

**审查不通过。** exp208 的 `GLOBAL_LOSS_SCALE=0.5` 在 GCN+PAA 路线下是一个 NO-OP——设置会被静默忽略，训练行为与 exp206 完全相同。必须重新审视实验目标后决定是否修改 loss 代码或取消实验。

---

**更新**: 经确认上述 Critical 问题属实后，若实验目标调整为其他有效变量，可重新审查。当前状态下此实验不应启动。

审查通过条件未满足。
