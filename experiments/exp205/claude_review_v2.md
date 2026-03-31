# exp205 审查 v2: GCN+PAA + STD-PR per-token SupCon (Dual Part Branch)

## 审查范围

- `experiments/exp205/design.md`
- `model/pose_backbone_model.py` — `__init__`, training path, test path 全部分支
- `loss/make_loss.py` — SupCon/CE/triplet 对 feat list 的处理
- `config/defaults.py` — 新增/缺失 config key
- `model/modules/skeleton_gcn.py` — return shape

---

## v1 修复验证

| v1 问题 | 修复状态 | 验证结果 |
|---------|---------|---------|
| C1: `__init__` 互斥 ValueError | 已删除 (行 154 不再有 raise) | 通过 — `__init__` 可同时初始化 structural_router + skeleton_head |
| C2: `POSE_DUAL_PART_BRANCH` 缺失 | 未加入 defaults.py，改为隐式触发 | 见 M1 新发现 |
| H1: `'str_cls_list' in dir()` | 替换为 `dual_branch_active` boolean | 通过 — 行 436 初始化 False, 行 462 设为 True |
| H2: SupCon 误包含 GCN feat | 使用 `kp_data['num_str_tokens']` + `feat[1:1+num_str]` | 部分通过 — 见 H1 新发现 |
| H3: `part_visibility` 缺失 | 在 dual path 中补充计算 (行 501-510) | 通过 — 与 STD-PR standalone 一致 |

---

## Critical — 必须修复

### C1: 训练路径 if/elif 分支逻辑导致 dual branch 永远不可达

**文件**: `model/pose_backbone_model.py`, 行 356 vs 行 432

训练时的分支结构为:
```python
if use_structural_routing and scene_heatmaps:    # 行 356
    ...
    return  # 行 424 (str_per_token=True) 或 行 430
elif use_skeleton_gcn and pose_dict:              # 行 432 — dual branch code 在此
    ...
```

当 exp205 同时设置 `POSE_STRUCTURAL_ROUTING=True` 和 `POSE_SKELETON_GCN=True` 时，行 356 的 `if` 条件先匹配，直接走 STD-PR standalone 路径并在行 424 return。**`elif` 块（行 432-516）中的全部 dual branch 代码永远不会执行。**

结果：模型运行时只有 STD-PR，没有 GCN。等价于 exp202b，不是 exp205 的设计意图。

**修复方案**: 必须重构 if/elif 分支。可行方案：
- 方案 A: 把 dual branch 判断提到最前面，作为独立的 `if` 分支:
  ```python
  if use_structural_routing and use_skeleton_gcn and scene_heatmaps and pose_dict:
      # dual branch code
  elif use_structural_routing and scene_heatmaps:
      # STD-PR standalone
  elif use_skeleton_gcn and pose_dict:
      # GCN standalone
  ```
- 方案 B: 在 `if use_structural_routing` 内部检测 `use_skeleton_gcn` 并分叉。

### C2: 测试路径同样存在 if/elif 分支问题

**文件**: `model/pose_backbone_model.py`, 行 528 vs 行 567

测试路径结构完全对称:
```python
if use_structural_routing and scene_heatmaps and pose_test_feat != 'global':  # 行 528
    ...
    gcn_feats = [str_feat]  # 行 566 — STD-PR pooled, 不含 GCN
elif use_skeleton_gcn and pose_dict and pose_test_feat != 'global':           # 行 567
    # Dual Part Branch test code 在此（行 571-585）
```

同 C1，当两者都启用时，行 528 匹配先执行，行 567 的 dual test path 不可达。测试特征只包含 global + STD-PR pooled，没有 GCN 特征。

**修复方案**: 与 C1 相同，需要在测试路径中也提前判断 dual branch 条件。

---

## High — 强烈建议修复

### H1: Additive CE 和 Triplet 仍使用 score[1:] / feat[1:] 包含 GCN token

**文件**: `loss/make_loss.py`

SupCon 已正确使用 `feat[1:1+num_str]`（行 174），但以下路径仍使用完整的 `[1:]`:

1. **行 194**: Additive CE — `score[1:]` 包含 GCN 的 cls score
   ```python
   part_ids = [ce_fn(s, target) for s in score[1:]]
   ```
   当 `POSE_STR_SUPCON_ADDITIVE=True` 时，CE 在 str_1..6 + gcn = 7 个 score 上计算。GCN 的 cls score 被当作 "part" 平均进 part_ce_avg。

2. **行 247**: Triplet — `feat[1:]` 包含 GCN feat
   ```python
   part_tris = [triplet(f, target, normalize_feature=use_norm)[0] for f in feat[1:]]
   ```
   GCN 的 768-D pooled feat 与 6 个 STD-PR tokens 混在一起计算 triplet loss。

这两处也需要用 `num_str_tokens` 来隔离。对于 GCN 部分，可以:
- CE: `score[1:1+num_str]` 用于 STD-PR, `score[1+num_str:]` 单独 CE
- Triplet: 同理分开

### H2: kp_data 未传入 `num_str_tokens` 在 non-SupCon 路径

**文件**: `loss/make_loss.py`, 行 203

当 `POSE_STR_SUPCON=False` 时（走 `else` 分支, 行 202-204），代码是:
```python
part_ids = [ce_fn(s, target) for s in score[1:]]
```
此路径不检查 `num_str_tokens`，GCN score 也被当作 part 平均。虽然 exp205 必然启用 SupCon，但如果有人想只用 CE 跑 dual branch（对照实验），此处会静默混入 GCN。

---

## Medium

### M1: `POSE_DUAL_PART_BRANCH` config flag 未加入 defaults.py

design.md 第 29 行写 "新 flag `POSE_DUAL_PART_BRANCH = False`"，但 defaults.py 中不存在。当前代码改为隐式通过 `use_structural_routing and use_skeleton_gcn` 判断 dual mode。

这本身不是 bug（只要 C1/C2 的分支逻辑修好），但与 design.md 不一致。建议二选一:
- 删除 design.md 中对 `POSE_DUAL_PART_BRANCH` 的提及，说明改为隐式触发
- 或添加显式 flag 以更清楚地控制行为

### M2: dual_branch_active 变量作用域

行 436 在 `elif self.use_skeleton_gcn` 块内初始化 `dual_branch_active = False`。由于 C1 的问题，这段代码当前不可达。但即使 C1 修复后，`dual_branch_active` 只在 `elif` 块内使用，作用域是安全的。此项无需额外修改。

### M3: Test path dual branch 特征拼接维度变化

行 584-585 将每个 structural token 逐一 append 到 `gcn_feats`:
```python
for k in range(structural_tokens.shape[1]):
    gcn_feats.append(structural_tokens[:, k])
```

`skeleton_head` 返回 `[skeleton_feat]`（1 个 768-D），再加 6 个 structural tokens → `gcn_feats` = 7 个 768-D features。加上 global，equal_concat 总维度 = 8*768 = 6144-D。

对比 GCN-only: 2*768 = 1536-D; STD-PR-only: 2*768 = 1536-D。
维度差异 4x 会显著改变距离计算的行为。v1 review M1 已提到，设计上这是有意为之还是需要重新考虑？建议在 design.md 中明确说明。

### M4: PLTD (Part-Level Token Dropout) 未在 dual branch 的 STD-PR block 中应用

行 373-391 在 STD-PR standalone 路径中实现了 PLTD，但 dual branch 的 STD-PR block（行 436-462）没有包含 PLTD 逻辑。如果 `POSE_STR_PART_DROP > 0`，standalone 和 dual 模式的训练行为不一致。

---

## Low

### L1: 注释 "SupCon operates on str_tok1..6" (行 496) 与实际行为一致

v1 的 L1 已随 H2 修复而解决。注释正确反映了 `feat[1:1+num_str]` 的行为。通过。

### L2: 冗余 `pose_dict is not None` 检查 (行 441)

仍存在但无害，与 v1 相同。

### L3: `_pg` 列表重复定义三次

行 396、504、543 都包含相同的 `_pg = [[0,1,2,3,4],[5,6,11,12],...]`。建议提取为类常量。不影响正确性。

---

## Backward Compatibility 验证

| 模式 | 是否受影响 |
|------|-----------|
| STD-PR standalone (`use_structural_routing=True, use_skeleton_gcn=False`) | 不受影响 — 行 356 if 分支不变 |
| GCN standalone (`use_structural_routing=False, use_skeleton_gcn=True`) | 不受影响 — 行 432 elif 分支, `dual_branch_active` 为 False |
| 无 part branch | 不受影响 — 行 518 return |
| PSG only | 不受影响 |

删除 ValueError (原行 154) 不影响已有实验: 过去没有实验同时启用两者，所以 ValueError 从未被触发。

---

## 结论

**审查未通过**

有 2 个 Critical 问题使 dual branch 完全不工作:
1. **训练路径** if/elif 分支导致 dual branch code 不可达 (C1)
2. **测试路径** 同样的 if/elif 问题 (C2)

这两个是 showstopper：模型能运行但行为等价于 STD-PR standalone，不是 exp205 设计的 dual branch。

另有 2 个 High 问题影响 loss 计算正确性:
1. Additive CE 和 Triplet 仍包含 GCN token (H1)
2. non-SupCon 路径未隔离 GCN (H2)

**必须修复 C1 + C2 后重新审查。** H1 也应同时修复以确保 loss 设计意图正确。
