# exp205 审查: GCN+PAA + STD-PR per-token SupCon (Dual Part Branch)

## 审查范围

- `experiments/exp205/design.md`
- `model/pose_backbone_model.py` (training path, test path, __init__)
- `loss/make_loss.py` (SupCon/CE/triplet on feat[1:])
- `config/defaults.py`
- `model/modules/skeleton_gcn.py` (return shape)

---

## Critical — 必须修复

### C1: `__init__` 中的互斥 ValueError 阻止模型构建

**文件**: `model/pose_backbone_model.py`, 行 154-155

```python
if self.use_structural_routing and self.use_skeleton_gcn:
    raise ValueError('POSE_STRUCTURAL_ROUTING and POSE_SKELETON_GCN cannot both be True')
```

exp205 要求两者同时为 True，但 `__init__` 会在模型构建时直接抛出异常。模型根本无法实例化。这是 showstopper。

**修复方案**: 引入 `POSE_DUAL_PART_BRANCH` flag (design.md 中提到但未实现)。当 dual branch 为 True 时跳过此检查，或直接移除此检查并让 forward 逻辑自行判断。同时需要确保 `__init__` 在 dual 模式下同时初始化 `structural_router` + `str_classifier` + `skeleton_head`。当前代码中 STD-PR 初始化在 `if self.use_structural_routing:` 块内 (行 156)，该块由于 ValueError 永远无法到达。

### C2: `config/defaults.py` 缺少 `POSE_DUAL_PART_BRANCH` 定义

design.md 提到新增 flag `POSE_DUAL_PART_BRANCH = False`，但 defaults.py 中不存在此定义。config YAML 中引用不存在的 key 会被静默忽略 (取决于 yacs 行为) 或报错。

---

## High — 强烈建议修复

### H1: `'str_cls_list' in dir()` 不可靠且不符合 Python 惯例

**文件**: `model/pose_backbone_model.py`, 行 493

`dir()` 用于检测局部变量是否被定义。虽然实测在 CPython 中可以工作，但这是 implementation-dependent 行为，不是 Python 语言规范保证的。`dir()` 的文档说 "attempts to produce the most relevant, rather than complete, information"。

**修复方案**: 使用显式 flag 变量:
```python
dual_str_active = False
# ... in the STD-PR block:
dual_str_active = True
# ... later:
if dual_str_active:
```

### H2: SupCon/CE/Triplet 误包含 GCN feature

**文件**: `loss/make_loss.py`, 行 172, 190, 243

当 feat = `[global, str_1..6, gcn]` (8 elements) 时:
- `feat[1:]` = `[str_1, str_2, ..., str_6, gcn]` (7 elements)
- SupCon 在全部 7 个 feature 上计算 (行 172)
- Additive CE 在全部 7 个 score 上计算 (行 190)
- Triplet 在全部 7 个 feature 上计算 (行 243)

design.md 明确写 "SupCon 在 str_1...6 上计算 (per-token)"，不包含 GCN。当前代码将 GCN pooled feature 也纳入 SupCon，改变了设计意图。

GCN 的 pooled skeleton feature 与 6 个 structural tokens 语义不同 -- GCN 是全局骨架特征，STD-PR tokens 是 part-level tokens。将 GCN 混入 per-token SupCon 可能不利于 SupCon 目标。

**修复方案**: 在 loss 中区分 str tokens 和 gcn token。可以在 kp_data 中记录 `'num_str_tokens': 6`，loss 中只对 `feat[1:1+num_str_tokens]` 计算 SupCon，对 GCN 单独计算 CE。或者调整返回顺序为 `[global, gcn, str_1..6]`，让 GCN 走不同的 loss 路径。

### H3: `part_visibility` 未在 dual path 中设置

**文件**: `model/pose_backbone_model.py`, 行 491-501

STD-PR standalone 路径 (行 421-423) 设置 `kp_data['part_visibility'] = part_w`，但 dual path (行 496-498) 没有设置 `part_visibility`。如果配置中启用了 `POSE_STR_SUPCON_VIS_WEIGHT`，loss 函数会 fallback 到均匀平均 (行 185-186)，不会报错，但会静默丢失 visibility weighting 功能。

此外 dual path 中根本没有计算 `part_w` (6-part heatmap visibility weights)，因为它没有运行 confidence-weighted pooling 代码块。

**修复方案**: 在 dual path 的 STD-PR block 中也计算 `part_w` 并写入 kp_data。

---

## Medium

### M1: Test path 中 structural tokens 未经 confidence-weighted pooling

**文件**: `model/pose_backbone_model.py`, 行 559-573

Dual branch test path 直接将每个 structural token 逐一 append 到 `gcn_feats`:
```python
for k in range(structural_tokens.shape[1]):
    gcn_feats.append(structural_tokens[:, k])
```

这意味着 test equal_concat 包含 `global + gcn + str_1..6` = 8 个 768-D features = 6144-D。而 STD-PR standalone test path (行 552-554) 使用 confidence-weighted pooled feature (`gcn_feats = [str_feat]`)，即 `global + pooled_part` = 2*768 = 1536-D。

训练时 SupCon 在 per-token 上操作 (正确)，但 test 时用 per-token concat vs pooled 是一个设计选择。8*768 = 6144-D 的特征维度会显著改变距离计算的性质，且可能不如 pooled feature。这不一定是 bug，但应在 design.md 中说明理由。

### M2: Backward compatibility -- GCN-only 模式不受影响

验证通过。当只启用 `use_skeleton_gcn` (不启用 `use_structural_routing`) 时:
- 行 436 的 `getattr(self, 'use_structural_routing', False)` 为 False
- STD-PR block 不执行
- 行 492 的 dual combine check 也为 False
- 直接走行 504 的 GCN-only return

此项通过，无需修改。

### M3: Memory overhead 评估

STD-PR 在 detached feature map 上运行 (行 433: `feat_map_detached = featmaps[-1].detach()`)，所以 STD-PR 的 forward 不会回传梯度到 backbone。额外的前向计算成本:
- StructuralRoutingLayer forward: cross-attention + optional self-attention
- 6 次 BN + classifier forward

这些在 detached tensor 上运行且不需要反向传播 (因为 STD-PR 的输出直接进 SupCon，梯度只经过 structural_router 和 str_classifier)。实际上 STD-PR forward 需要梯度 (用于 SupCon 优化 router 参数)，但不影响 backbone。内存增量估计 ~200-500MB，可接受。

---

## Low

### L1: 注释与代码不完全一致

行 494-495 注释 "SupCon operates on str_tok1..6" 但代码实际上 SupCon 操作在 `feat[1:]` 上 (包含 GCN)。如果 H2 修复后此注释是准确的，否则应更新。

### L2: 冗余的 `pose_dict is not None` 检查

行 440: `kp_p0 = pose_dict['keypoints'][:, 0] if pose_dict is not None else None`

此代码在 `elif self.use_skeleton_gcn and pose_dict is not None:` (行 432) 的分支内，`pose_dict` 已经保证不为 None。检查冗余但无害。

---

## 结论

**审查未通过**

有 2 个 Critical 问题阻止模型运行:
1. `__init__` 互斥 ValueError 使模型无法实例化 (C1)
2. `POSE_DUAL_PART_BRANCH` config 未定义 (C2)

以及 3 个 High 问题影响训练正确性:
1. `'str_cls_list' in dir()` 不可靠 (H1)
2. SupCon/CE/triplet 误包含 GCN feature (H2)
3. `part_visibility` 未在 dual path 设置 (H3)

必须修复 C1+C2+H1+H2 后重新审查。H3 视是否使用 vis_weight 决定优先级。
