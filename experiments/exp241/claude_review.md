# exp241 Claude Review: PPA + GCN 双分支

## 审查范围

a. design.md 合理性
b. 训练路径：PPA non-detached + GCN detached 梯度隔离
c. 输出形状：loss 函数对变长列表的处理
d. 测试路径：PPA + GCN 特征拼接
e. OA-SD 兼容性：teacher model 同结构

---

## a. design.md 合理性

设计动机清晰：PPA 提供 end-to-end backbone 训练信号，GCN 在更好的 backbone 上采样提供互补特征。两者功能不重叠：PPA 改善 backbone 表征，GCN 提供 skeleton-guided 局部特征。对照组明确（exp237 PPA-only, exp191 GCN-only）。

**注意**：这本质上是一个组合实验（PPA + GCN），不是全新创新。但因为 PPA 和 GCN 在梯度路径上完全正交（PPA non-detached, GCN detached），组合有合理的技术依据，不是简单堆叠。

## b. 训练路径 — 梯度隔离

**代码位置**: `pose_backbone_model.py` 行 480-499

```python
elif getattr(self, 'use_ppa', False) and scene_heatmaps is not None:
    ppa_cls_scores, ppa_feats, ppa_data = self.part_assignment_head(
        featmaps[-1], scene_heatmaps, return_cls=True)  # non-detached ✓
    ...
    if self.use_skeleton_gcn and pose_dict is not None:
        feat_map_detached = featmaps[-1].detach()  # detached ✓
        gcn_cls_scores, gcn_feats, gcn_data = self.skeleton_head(
            feat_map_detached, pose_dict, ...)
```

- PPA 接收 `featmaps[-1]`（non-detached）：梯度流回 backbone ✓
- GCN 接收 `featmaps[-1].detach()`：梯度只到 GCN 自身 ✓
- Stage2 feat 也正确 detach：`featmaps[-2].detach()` ✓
- 梯度隔离正确，两个分支不会互相干扰

## c. 输出形状与 Loss 处理

训练输出结构：
- `score = [global_cls] + [ppa_pooled, ppa_p1..p5] + [gcn_cls]` = 8 个元素
- `feat = [global_feat] + [ppa_pooled, ppa_p1..p5] + [gcn_skeleton]` = 8 个元素

Loss 函数（`make_loss.py`）处理：
- CE: `part_ids = [ce_fn(s, target) for s in score[1:]]` → 7 个 part CE 取平均 ✓
- Triplet: `part_tris = [triplet(f, target)[0] for f in feat[1:]]` → 7 个 part triplet 取平均 ✓
- PPA assign_loss: 通过 kp_data 单独加上 ✓
- `POSE_PART_WEIGHT` 控制 global/part 比例 ✓

**无问题**：变长列表被正确迭代处理，不依赖固定长度。

## d. 测试路径

**代码位置**: `pose_backbone_model.py` 行 645-657

```python
_, ppa_feats, aux_data = self.part_assignment_head(
    featmaps[-1], scene_heatmaps, return_cls=False)
gcn_feats = ppa_feats  # [pooled, part1..part5]
if self.use_skeleton_gcn and pose_dict is not None:
    _, gcn_only_feats, gcn_aux = self.skeleton_head(
        featmaps[-1], pose_dict, return_cls=False)
    gcn_feats = ppa_feats + gcn_only_feats
```

- PPA 返回 6 个特征，GCN 返回 1 个 → 合计 7 个 ✓
- equal_concat 模式：`[g_norm] + p_norm` = 1 + 7 = 8 * 768 = 6144 维 ✓
- 测试时 GCN 接收 `featmaps[-1]`（非 detach），但测试时无梯度，无影响 ✓
- MaxSim 路径：`aux_data` 中的 `kp_feats` 来自 PPA（5 parts），`gcn_kp_feats` 来自 GCN ✓

## e. OA-SD 兼容性

Teacher（EMA）与 student 是同一模型类，forward 输出结构相同。

OA-SD distillation 路径（行 756-764）：
- `zip(feat, teacher_feat)` 逐元素配对 → 两边都是 8 个元素 ✓
- Global-only 模式：`feat[0]` / `teacher_feat[0]` ✓
- Relational distillation：`feat[0]` / `teacher_feat[0]` ✓

**无问题**：student 和 teacher 的输出列表长度一致。

## 潜在注意点（非 blocking）

1. **特征维度较大**：equal_concat 产生 6144 维特征（8 * 768），比 PPA-only（4608 维）或 GCN-only（1536 维）都大。距离计算开销增加但不影响正确性。
2. **Loss 权重平衡**：7 个 part loss 取平均，其中 PPA 的 6 个 part 占 6/7 权重，GCN 的 1 个 skeleton 占 1/7 权重。PPA 部分在 loss 中天然权重更大，这可能是合理的（PPA 是主训练信号）。
3. **kp_data 合并**：PPA 的 kp_data 为基础，GCN 的 `kp_feats` 和 `kp_weights` 以 `gcn_` 前缀存入。Processor 中 PKC 等模块使用 `kp_data['kp_feats']`（PPA 的），不会误用 GCN 的 ✓

## 结论

代码逻辑正确，梯度隔离清晰，loss 处理无误，测试路径输出一致，OA-SD 兼容。

**审查通过**
