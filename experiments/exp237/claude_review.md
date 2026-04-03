# exp237 PPA 审查报告

审查人: Claude Opus 4.6
日期: 2026-04-03

## 审查范围

a. `experiments/exp237/design.md` — 合理性、创新门槛
b. `model/modules/part_assignment_head.py` — 逐行审查
c. `model/pose_backbone_model.py` — PPA 集成路径（训练 + 测试）
d. `processor/processor.py` — assign_loss 集成
e. `config/defaults.py` — 新 config 键
f. 梯度流分析 — 端到端训练干扰风险

---

## A. 设计文档审查

**动机清晰**: 236 个实验的系统总结证明 detached feature 上的所有操作都无法改善最终结果。只有改变 backbone 本身行为的方法（PSG, OA-SD）有效。PPA 尝试用 learnable part-assignment head 取代 detached GCN sampling，让 part loss 梯度端到端流过 backbone。

**创新门槛**:
1. 问题层面: 从"backbone 输出上做 post-hoc pooling" 变为 "backbone 输出上做 learnable part assignment with pose supervision"。满足。
2. 机制层面: 用 pose heatmap 做 soft assignment supervision，不同于 KPR 的 parsing label。有差异但差异幅度一般。部分满足。
3. 证据层面: 可直接对比 detached GCN (exp191)。消融: assign_loss 权重, part 数量, detach vs non-detach。满足。

**总评**: 满足创新门槛 2/3 条。方向正确——这是 236 个实验后最合理的下一步。

---

## B. `model/modules/part_assignment_head.py` 逐行审查

### B1. `__init__` (L41-74) — 通过

- `part_proj = nn.Linear(768, 6)`: 正确，K+1 = 5 parts + background。
- **零初始化** (L52-53): `nn.init.zeros_` 使初始 softmax 输出均匀分布 (1/6 each)。这是安全的，等价于 identity start (均匀加权平均 ≈ GAP)。
- `part_bns`: 每个 part 独立 BN1d，bias.requires_grad=False。标准 BNNeck 做法。正确。
- `part_classifiers`: 5 个独立 Linear(768, num_classes, bias=False)。正确。
- `pooled_bn` + `pooled_classifier`: 用于 visibility-weighted 聚合特征的分类。正确。

### B2. `_heatmaps_to_part_labels` (L76-111) — 有问题，见下方

- L91-92: `F.interpolate` 将 heatmap 从 (B,17,H,W) 缩到 (B,17,fH,fW)=(B,17,12,4)。正确。
- L95-98: 用 `torch.max` 逐 keypoint 聚合到 5 个 part。正确的 max-merge 策略。
- L102: `max_val, max_part = part_hm.max(dim=1)` — 找每个空间位置最强的 part。正确。
- L105: background threshold 计算: `threshold = max_val.flatten(1).mean(dim=1, ...)` — 这是 per-sample 的均值。`* 0.3` 给出较低阈值。
- **[Medium] L105 shape**: `max_val.flatten(1).mean(dim=1, keepdim=True)` → shape (B, 1)，`.unsqueeze(-1)` → (B, 1, 1)。然后 `max_val < threshold * 0.3` 比较 (B, fH, fW) vs (B, 1, 1) — broadcasting 正确。
- L108-109: `labels[is_bg] = self.num_parts` (=5, background)。正确，CE 的 target 范围是 [0, 5]，part_proj 输出 6 个 logit。
- L111: `labels.flatten(1)` → (B, 48)。正确。

**无 bug，逻辑正确。**

### B3. `forward` (L113-186)

- L126: `tokens = feat_map.flatten(2).transpose(1, 2)` — (B, C, 12, 4) → (B, 48, 768)。正确。
- L129-130: logits (B, 48, 6), probs (B, 48, 6)。正确。

**Assignment loss (L133-139)**:
- L137: `logits.reshape(-1, self.num_labels)` → (B*48, 6)
- L138: `gt_labels.reshape(-1)` → (B*48,)
- `F.cross_entropy` 期望 (N, C) vs (N,)。正确。

**Per-part pooling (L141-153)**:
- L145: `weights = probs[:, :, k].unsqueeze(-1)` → (B, 48, 1)。正确。
- L146: `w_sum = weights.sum(dim=1).clamp(min=1e-6)` → (B, 1)。
- L147: `(tokens * weights).sum(dim=1)` → (B, 768)，除以 (B, 1) → (B, 768)。正确。无 NaN 风险（clamp 保护）。

**Visibility (L156)**:
- `probs[:, :, :5].max(dim=1)[0]` → (B, 5)。对每个 part，取所有 48 个 token 中最大的 assignment probability。这是一个合理的 visibility proxy。

**Pooled feature (L157-159)**:
- `vis_weights` 归一化后用于加权平均 5 个 part feature。正确。
- `sum(vis_weights[:, k:k+1] * part_feats[k] ...)` — 切片 (B, 1) * (B, 768) → (B, 768)。正确。

**Aux data (L162-175)**:
- `kp_feats`: (B, K=5, 768) — 供 MaxSim 使用。正确。
- `kp_weights`: visibility (B, 5)。正确。
- `assign_entropy` 计算: `-(probs * (probs + 1e-8).log()).sum(dim=-1).mean()` — 先对 K+1 维求和得 token-level entropy，再全局平均。正确。

**Return format (L177-186)**:
- Training: `([pooled_cls, part1_cls, ..., part5_cls], [pooled_feat, part1_feat, ..., part5_feat], aux_data)` — 6 个 cls scores，6 个 features。
- Test (return_cls=False): `(None, [pooled_feat, part1_feat, ..., part5_feat], aux_data)`。正确。

**无 bug。**

---

## C. `model/pose_backbone_model.py` PPA 集成

### C1. 初始化 (L130-141) — 通过

- `self.use_ppa = cfg.MODEL.POSE_PPA`。正确。
- `self.part_assignment_head` 构造使用 `self.in_planes` (=768 for Swin-Tiny)。正确。
- `self.pose_test_feat` 设置。正确。

### C2. 训练路径 (L480-485) — 通过

```python
elif getattr(self, 'use_ppa', False) and scene_heatmaps is not None:
    ppa_cls_scores, ppa_feats, ppa_data = self.part_assignment_head(
        featmaps[-1], scene_heatmaps, return_cls=True)
    kp_data = ppa_data
    return [cls_score] + ppa_cls_scores, [global_feat] + ppa_feats, featmaps, None, kp_data
```

- `featmaps[-1]` 是 **NOT detached** — 这是 PPA 的核心设计。正确，有意为之。
- 返回格式: `([global_cls, pooled_cls, part1_cls, ..., part5_cls], [global_feat, pooled_feat, part1, ..., part5], ...)` — 7 个元素。
- `kp_data = ppa_data` 传递 assign_loss 到 processor。正确。

**[注意] PPA 与 GCN 互斥**: elif 链中 PPA 在 GCN 之前。如果同时启用 PPA 和 GCN，只有 PPA 生效。设计合理。

### C3. 测试路径 (L628-633) — 通过

```python
if getattr(self, 'use_ppa', False) and scene_heatmaps is not None and \
        getattr(self, 'pose_test_feat', 'global') != 'global':
    _, ppa_feats, aux_data = self.part_assignment_head(
        featmaps[-1], scene_heatmaps, return_cls=False)
    gcn_feats = ppa_feats
```

- `return_cls=False` 跳过 BN+classifier，只返回 features。正确。
- `aux_data` 包含 `kp_feats` 和 `kp_weights`，供 MaxSim test path 使用。
- `gcn_feats = ppa_feats` 然后由后续的 `equal_concat`/`maxsim` 等 test mode 处理。正确。

**[注意]** 当 `scene_heatmaps is None`（无 pose 数据的样本），PPA 不执行，fallback 到 global-only feature。这是安全的 fallback。

### C4. OA-SD 兼容性

**Teacher 也会运行 PPA**: OA-SD teacher 在 `train()` mode (line 714) + `no_grad()` 下执行 forward。由于 `self.training=True`，PPA 的训练路径会被触发。Teacher 产生相同结构的输出: `[global_cls, pooled_cls, part1_cls, ..., part5_cls]`。

- Teacher 的 `assign_loss` 会被计算但在 `no_grad()` 下，不产生梯度。无害。
- Distillation: `zip(student_feat, teacher_feat)` 逐元素匹配。Student 7 features vs Teacher 7 features — 结构匹配。正确。
- Teacher's PPA 也使用 non-detached features，但在 no_grad 下无影响。

**无 bug。**

---

## D. `processor/processor.py` assign_loss 集成 (L893-903) — 通过

```python
if ppa_enabled and kp_data is not None and 'assign_loss' in kp_data:
    assign_weight = float(getattr(cfg.MODEL, 'POSE_PPA_ASSIGN_WEIGHT', 0.5))
    assign_loss = kp_data['assign_loss']
    loss = loss + assign_weight * assign_loss
```

- `kp_data` 来自 model output 的第 5 个元素 (line 513)。正确。
- `assign_loss` 是 scalar tensor。正确。
- `assign_weight = 0.5` 默认值。添加到总 loss。正确。
- 日志: `ppa_assign`, `ppa_bg_ratio`, `ppa_entropy` 被记录。充分。

**注意**: assign_loss 在 loss_fn() 之后额外添加，所以它与 list-loss 路径中的 ID/triplet loss 是叠加关系，不会冲突。

---

## E. `config/defaults.py` — 通过

```python
_C.MODEL.POSE_PPA = False                 # 安全默认
_C.MODEL.POSE_PPA_NUM_PARTS = 5           # 合理
_C.MODEL.POSE_PPA_ASSIGN_WEIGHT = 0.5     # 合理初始值
```

所有默认值为 False/保守值，不影响已有实验。

---

## F. 梯度流分析

**PPA 的梯度来源 (全部流过 backbone)**:
1. **Part ID loss** (CE): 5 个 part classifier 的 CE loss → 通过 `part_feats[k]` → 通过 softmax weighted pooling → 通过 `tokens = feat_map.flatten(2).transpose(1,2)` → 通过 `feat_map = featmaps[-1]` (NOT detached) → backbone。
2. **Part triplet loss**: 同上。
3. **Assignment CE loss**: `F.cross_entropy(logits, gt_labels)` → 通过 `logits = self.part_proj(tokens)` → 通过 `tokens` → backbone。
4. **Global ID + triplet**: 不变，仍然流过 backbone。

**vs GSPB (已失败)**:
- GSPB 用 `_gs * (featmaps[-1] - featmaps[-1].detach())` 做 scaled gradient — 效果是缩放 part loss 到 backbone 的梯度。
- PPA 的梯度是 **unscaled**，所有 part loss 全量流过 backbone。

**[Medium] 干扰风险**:
- 端到端梯度意味着 Part ID loss + Part triplet loss + Assignment CE 都优化 backbone。
- 结合 Global ID + Global triplet + OA-SD distillation，backbone 同时接收 6-7 种 loss 的梯度。
- GSPB 在 scale=0.05 时都有 late interference (exp229/230)。PPA 相当于 scale=1.0。
- **但 PPA 和 GSPB 不同**: PPA 用的是 softmax weighted average pooling (smooth)，而 GCN 用的是 bilinear grid_sample (sparse, 只采 17 个点)。Softmax pooling 的梯度分布更均匀，可能干扰更温和。
- **这是实验需要验证的假设，不是代码 bug。** 设计文档已预见 "softmax CE 仍然干扰 backbone ID loss" 的失败模式。

---

## 发现的问题

### [Medium] M1: pooled_feat 和 part_feats 特征空间不一致

`pooled_feat` 是 5 个 part_feats 的 visibility-weighted 平均。在 loss_fn 中，`score[0]` 是 `global_cls`，`score[1]` 是 `pooled_cls`，`score[2:]` 是 `part_cls`。所以 `pooled_feat` 被 `pooled_classifier` 分类，但它是 part_feats 的加权平均——这意味着 pooled_classifier 和 part_classifiers 看到的是高度相关但不完全相同的特征分布。这不会导致 bug，但增加了一个冗余的 classifier head。

**建议**: 可以监控 pooled 分支 vs part 分支的 ID accuracy 差异来判断是否冗余。不阻塞。

### [Low] L1: Test-time assign_loss 仍被计算（浪费）

在 test path (return_cls=False)，`forward` 中 `self.training` 为 False，所以 `assign_loss` 为 `torch.tensor(0.0)`。这是无害的小开销，但可以优化。不阻塞。

### [Low] L2: 缺少配置文件

没有在 `configs/` 下找到 exp237 的 YAML 配置文件。需要在训练前创建。不阻塞。

---

## 审查结论

代码逻辑正确，无 Critical / High 级别问题。数据流清晰:
- 训练: feat_map (B,768,12,4) → tokens (B,48,768) → logits (B,48,6) → probs → weighted pool → 5 part feats + 1 pooled feat → 6 个 CE + 6 个 triplet + 1 个 assign CE
- 测试: 同上但跳过 classifier，返回 features 用于 equal_concat/maxsim
- OA-SD: teacher 结构匹配，distillation 安全
- 梯度端到端流过 backbone，这是 PPA 的核心设计

所有发现的 Medium/Low 问题均非阻塞。

**审查通过**
