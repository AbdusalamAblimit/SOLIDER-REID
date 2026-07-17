# exp233 审查报告: Per-Body-Part Independent Training (KPR-Inspired)

**审查日期**: 2026-04-03
**审查范围**: design.md, config/defaults.py, model/modules/skeleton_gcn.py, model/pose_backbone_model.py, loss/make_loss.py, processor/processor.py

---

## a. design.md — 合理性与单变量

**判定: 通过 (Medium 级备注)**

- 动机清晰: KPR ECCV 2024 的核心差异确实在于 per-body-part 独立训练, 在现有 GCN 框架上实现此功能是合理的消融实验。
- 单变量: 只增加 POSE_GCN_PER_PART=True, 其余保持不变。对照 exp191 (OA-SD + pooled GCN)。满足单变量原则。
- 创新门槛: 这是一个 KPR 组件的消融/移植实验, 本身不构成独立创新, 但作为 "验证 per-part training 是否在我们框架中有效" 的实验是合理的。

**Medium**: design.md 中参数量估算 "6 x 1.5K = 9K extra params" 严重低估。实际每个 part 新增 BN(768) + Linear(768, 702) = 1,536 + 539,136 = ~540K params。6 parts 总计 ~3.24M extra params, 是声称值的 360 倍。这不影响代码正确性, 但文档应修正。

## b. 代码审查

### config/defaults.py (line 197)

**判定: 通过**

```python
_C.MODEL.POSE_GCN_PER_PART = False
```

默认值 False, 不影响任何已有实验。安全。

### model/modules/skeleton_gcn.py — BODY_PART_GROUPS 常量 (lines 44-51)

**判定: 通过 (Low 级备注)**

分组定义:
- head: [0,1,2,3,4] — nose, eyes, ears (5 kps)
- torso: [5,6,11,12] — shoulders, hips (4 kps)
- left_arm: [5,7,9] — l_shoulder, l_elbow, l_wrist (3 kps)
- right_arm: [6,8,10] — r_shoulder, r_elbow, r_wrist (3 kps)
- left_leg: [11,13,15] — l_hip, l_knee, l_ankle (3 kps)
- right_leg: [12,14,16] — r_hip, r_knee, r_ankle (3 kps)

**Low**: 关键点 5, 6, 11, 12 (肩膀和髋部) 同时出现在 torso 和 arm/leg 组中。这是 KPR 论文中的标准做法 (肩膀是手臂的起点, 也是躯干的一部分), 在训练时每个组独立 pool 不会引起梯度问题 (因为是从 detached features 上操作), 但需要注意这意味着这些关键点的特征会被多个分类器同时训练。这是设计意图, 非 bug。

### model/modules/skeleton_gcn.py — SkeletonGCNHead.__init__ (lines 455-468)

**判定: 通过**

```python
if self.per_part:
    num_parts = len(BODY_PART_GROUPS)  # = 6
    self.part_bns = nn.ModuleList([nn.BatchNorm1d(feat_dim) for _ in range(num_parts)])
    for bn in self.part_bns:
        bn.bias.requires_grad_(False)
    self.part_classifiers = nn.ModuleList([
        nn.Linear(feat_dim, num_classes, bias=False) for _ in range(num_parts)])
```

- 使用 ModuleList 正确注册参数, 确保进入 optimizer
- BN bias requires_grad=False 与全局 BN 一致
- BN 初始化在 line 466-468 正确调用
- AMP 安全: BN1d 和 Linear 都是标准模块

### model/modules/skeleton_gcn.py — forward per_part training (lines 1014-1028)

**判定: 通过**

```python
if self.per_part:
    part_cls_scores = []
    part_feats = []
    for i, group_indices in enumerate(BODY_PART_GROUPS):
        group_feats = kp_feats_enhanced[:, group_indices, :]  # (B, G, C)
        group_weights = kp_weights[:, group_indices].clamp(min=1e-6).unsqueeze(-1)  # (B, G, 1)
        part_feat = (group_feats * group_weights).sum(dim=1) / \
                    group_weights.sum(dim=1).clamp(min=1e-6)  # (B, C)
        part_feat_bn = self.part_bns[i](part_feat)
        part_cls = self.part_classifiers[i](part_feat_bn)
        part_cls_scores.append(part_cls)
        part_feats.append(part_feat)
    return [cls_score] + part_cls_scores, [skeleton_feat] + part_feats, aux_data
```

**Shape 验证**:
- `kp_feats_enhanced`: (B, 17, C=768). `group_indices` 如 [0,1,2,3,4] → `group_feats`: (B, 5, 768). 正确。
- `kp_weights`: (B, 17). `kp_weights[:, group_indices]`: (B, 5). `.clamp().unsqueeze(-1)`: (B, 5, 1). 正确。
- weighted pool → (B, C). BN1d(768) 输入 (B, 768). 正确。
- 返回: `[cls_score] + part_cls_scores` = 1 + 6 = 7 elements. `[skeleton_feat] + part_feats` = 1 + 6 = 7 elements。

注意: 这里的 `cls_score` 是 **pooled skeleton** 的分类分数 (不是 global), `skeleton_feat` 是 **pooled skeleton** 的特征。这些被 pose_backbone_model.py 包裹后在外层再加上 global。

### model/modules/skeleton_gcn.py — forward per_part test (lines 1032-1043)

**判定: 通过**

```python
if self.per_part:
    part_feats = []
    for group_indices in BODY_PART_GROUPS:
        group_feats = kp_feats_enhanced[:, group_indices, :]
        group_weights = kp_weights[:, group_indices].clamp(min=1e-6).unsqueeze(-1)
        part_feat = (group_feats * group_weights).sum(dim=1) / \
                    group_weights.sum(dim=1).clamp(min=1e-6)
        part_feats.append(part_feat)
    return None, [skeleton_feat_out] + part_feats, aux_data
```

- Test 返回: `[skeleton_feat_out, part1, ..., part6]` = 7 features. 注意这里第一个元素是 `skeleton_feat_out` (可能经过 PKE 处理), 后面是 raw per-part features (未经 BN 处理)。这与 train 时一致 (train 时返回 `skeleton_feat` 即未 BN 版本, BN 后的 `feat_bn` 在 aux_data 中)。
- 在 pose_backbone_model.py 的 test 路径中, 这些会被 equal_concat 串接: global + skeleton + 6 parts = 8 * 768 = 6144 维。比之前的 2 * 768 = 1536 大 4 倍, 但功能正确。

### model/pose_backbone_model.py (line 151)

**判定: 通过**

```python
per_part=getattr(cfg.MODEL, 'POSE_GCN_PER_PART', False),
```

Pass-through 正确, 使用 getattr 确保向后兼容。

## c. Shape 正确性

**判定: 通过**

完整数据流:
1. `kp_feats_enhanced`: (B, 17, 768) — GCN 增强后
2. `group_indices` e.g. [0,1,2,3,4] → `kp_feats_enhanced[:, [0,1,2,3,4], :]`: (B, 5, 768) ✓
3. `kp_weights[:, [0,1,2,3,4]]`: (B, 5) → `.clamp().unsqueeze(-1)`: (B, 5, 1) ✓
4. Weighted sum → (B, 768) ✓
5. BN1d(768) input: (B, 768) ✓
6. Linear(768, 702) input: (B, 768) → output: (B, 702) ✓

所有 group_indices 在 [0, 16] 范围内, 不会越界。

## d. Loss 兼容性

**判定: 通过**

make_loss.py 处理 `score[1:]` 循环, 不限制长度:
- `score[1:]` = [skeleton_cls, part1_cls, ..., part6_cls] = 7 elements → 7 个独立 CE loss, 取平均
- `feat[1:]` = [skeleton_feat, part1, ..., part6] = 7 elements → 7 个独立 triplet loss, 取平均
- `len(feat) = 8 > 3` → triplet 使用 L2 normalization (per-token mode)。这是合理的, 因为 per-part features 维度相同。
- Global/Part loss 权重分配: w_g=0.5 (global CE), w_p=0.5 (averaged over 7 part CEs)。每个 part classifier 实际梯度权重 = 0.5/7 ≈ 0.071。

OA-SD 兼容性: teacher/student 都产生 8 元素 feat 列表, `zip(feat, teacher_feat)` 逐元素匹配, 正确。

## e. Test-time 行为

**判定: 通过**

- `equal_concat` 模式: 全部 L2-normalize 后 concat → 8*768 = 6144 维。正确但维度较大。
- `maxsim` 模式: 使用 `aux_data['kp_feats']` (17 个原始关键点特征), 不受 per_part 影响。正确。
- `concat_scaled` 模式: `[test_feat] + [f * scale for f in gcn_feats]`, scale = 1/7。正确。

## f. 默认值安全

**判定: 通过**

`POSE_GCN_PER_PART` 默认为 False。`self.per_part = False` 时:
- `__init__` 不创建 part_bns 和 part_classifiers
- `forward` 走 `else` 分支, 返回 `[cls_score]` 和 `[skeleton_feat]`, 与改动前完全一致
- 不影响任何已有实验的可复现性

---

## 额外检查

### Optimizer 参数注册
`part_bns` 和 `part_classifiers` 使用 `nn.ModuleList` 注册, 会自动进入 `model.parameters()` → optimizer 正确优化。无遗漏。

### AMP 安全性
所有新增组件 (BN1d, Linear) 都是 PyTorch 原生模块, AMP autocast 完全支持。

### 日志充分性
- make_loss.py 记录 `id_part` (7个 CE 的平均) 和 `tri_part` (7个 triplet 的平均)。无法区分各 body part 的单独 loss, 但对监控整体趋势足够。
- SkeletonGCNHead 的 print 语句 (line 462) 会在初始化时输出 `[GCN] Per-body-part training enabled: 6 parts`, 可确认配置生效。

---

## 汇总

| 级别 | 数量 | 详情 |
|------|------|------|
| Critical | 0 | — |
| High | 0 | — |
| Medium | 1 | design.md 参数量估算错误 (声称 9K, 实际 ~3.24M) |
| Low | 1 | BODY_PART_GROUPS 中关键点有重叠 (设计意图, 非 bug) |

Medium 级问题为文档准确性问题, 不影响代码运行。建议在 design.md 中修正参数量, 但不阻塞训练启动。

## 结论

**审查通过**。代码实现正确, shape 验证通过, loss 兼容, 默认值安全, AMP 无风险。唯一需要注意的是 test-time feature 维度从 1536 增大到 6144 (equal_concat 模式), 可能略微增加评估时间, 但不影响功能。
