# exp205 审查 v3: GCN+PAA + STD-PR per-token SupCon (Dual Part Branch)

## 审查范围

- `model/pose_backbone_model.py` — 训练路径 if/elif 分支 (line 356-517), 测试路径 (line 529-586)
- `loss/make_loss.py` — SupCon/CE/triplet 对 feat list 的处理 (line 160-248)
- v2 中 C1/C2 修复验证 (`not self.use_skeleton_gcn` guard)
- v1 修复持续验证 (dual_branch_active, num_str_tokens, part_visibility)
- 向后兼容性: STD-PR-only, GCN-only, no-part-branch 三种模式

---

## v2 Critical 修复验证

### C1 (训练路径 if/elif 分支): 已修复

**修复内容**: Line 356-357 现在是:
```python
if getattr(self, 'use_structural_routing', False) and scene_heatmaps is not None \
        and not self.use_skeleton_gcn:
```

四种组合的路径分析:

| use_structural_routing | use_skeleton_gcn | Line 356 if | Line 433 elif | 结果 |
|---|---|---|---|---|
| True | False | True (进入) | 跳过 | STD-PR standalone -- 正确 |
| False | True | False (跳过) | True (进入), dual_branch_active=False | GCN standalone -- 正确 |
| True | True | False (跳过, `not self.use_skeleton_gcn` = False) | True (进入), dual_branch_active=True | Dual branch -- 正确 |
| False | False | False | False | Line 519: 返回 scalar score/feat -- 正确 |

**结论: 通过。** 当两者同时启用时, `not self.use_skeleton_gcn` 使第一个 `if` 为 False, 流程正确落入 `elif` 的 dual branch 代码。

### C2 (测试路径 if/elif 分支): 已修复

**修复内容**: Line 529-530 现在是:
```python
if getattr(self, 'use_structural_routing', False) and scene_heatmaps is not None and \
        getattr(self, 'pose_test_feat', 'global') != 'global' and not self.use_skeleton_gcn:
```

四种组合的路径分析:

| use_structural_routing | use_skeleton_gcn | Line 529 if | Line 568 elif | 结果 |
|---|---|---|---|---|
| True | False | True | 跳过 | STD-PR test path -- 正确 |
| False | True | False | True, 进入 GCN test + no dual add | GCN test -- 正确 |
| True | True | False (`not self.use_skeleton_gcn` = False) | True, line 573 内部 if 检测 structural_routing → 执行 dual test | Dual test: GCN feats + STD-PR tokens 全部拼接 -- 正确 |
| False | False | False | False | 无 part branch, 纯 global test -- 正确 |

**结论: 通过。** 测试路径与训练路径对称, dual 模式正确进入 `elif` 块并在内部检测 structural_routing 追加 tokens。

---

## v1 修复持续验证

| v1 问题 | 当前状态 |
|---------|---------|
| `__init__` 不再 raise ValueError (原行 154) | 通过 — Line 154 现为注释说明 "Dual Part Branch: both can now be enabled simultaneously" |
| `dual_branch_active` boolean 替换 `'str_cls_list' in dir()` | 通过 — Line 437 初始化 False, line 463 设为 True |
| `kp_data['num_str_tokens']` 传递 | 通过 — Line 501 设置, loss/make_loss.py line 173 读取 |
| `part_visibility` 在 dual path 计算 | 通过 — Line 503-511 与 standalone 路径 (line 394-405) 逻辑一致 |

---

## v2 High 问题复查

### H1: Additive CE 和 Triplet 仍使用 score[1:] / feat[1:] 包含 GCN token

**状态: 未修复, 但影响有限**

1. **Additive CE (line 194)**: `score[1:]` = str_tok_1..6 + gcn_cls = 7 个 score。GCN 的 cls score 被当作第 7 个 "part" 平均进 part_ce_avg。
2. **Triplet (line 247)**: `feat[1:]` = str_tok_1..6 + gcn_feat = 7 个 feat。GCN 的 768-D pooled feat 与 6 个 STD-PR tokens 一起计算 triplet。

影响评估:
- CE: 7 个 score 平均 vs 6 个平均, GCN 本身也需要 CE loss 来训练分类器, 所以多包含一个 GCN CE 并非错误, 只是 weight 不同 (1/7 vs 如果单独就是 1/1)。不会导致训练崩溃。
- Triplet: 同理, GCN 被当作第 7 个 part 做 triplet, 有一定训练效果, 不是 bug。
- `use_norm = len(feat) > 3` → 7 > 3 → True → 所有 per-token feat 做 L2-norm 再 triplet, 包括 GCN。这是合理的。

**严重程度: Medium (降级)**。功能上不会出错, 但 loss 权重分配与设计意图略有偏差 (GCN 同时被 SupCon 排除但被 CE/triplet 包含)。如果要严格控制, 应修改, 但不阻塞训练。

### H2: non-SupCon 路径 score[1:]

**状态: 不影响 exp205**。exp205 必然启用 SupCon, 此路径 (line 202-204) 不会被触发。保留为 Low。

---

## 数据流完整验证 (dual branch, 训练)

1. backbone → `global_feat` (768-D), `featmaps[-1]` (B, 768, 12, 4)
2. `feat_map_detached = featmaps[-1].detach()` — STD-PR 和 GCN 都用 detached feature map
3. STD-PR (line 438-462): `structural_router` → `structural_tokens` (B, 6, 768) → per-token CE/triplet features → `str_cls_list` (6 items), `str_feat_list` (6 items)
4. GCN (line 465-466): `skeleton_head` → `gcn_cls_scores` (list), `gcn_feats` (list)
5. 合并 (line 512-514): `[cls_score] + str_cls_list + gcn_cls_scores`, `[global_feat] + str_feat_list + gcn_feats`
6. Loss: SupCon on `feat[1:1+6]` = STD-PR tokens only. CE on `score[1:]` = all 7. Triplet on `feat[1:]` = all 7.

**通过。** 数据流正确, STD-PR 和 GCN 各自独立计算, 合并为统一的 score/feat list 返回。

---

## 数据流完整验证 (dual branch, 测试)

1. Line 568-571: 进入 `elif self.use_skeleton_gcn` 块, 运行 `skeleton_head` → `gcn_feats` (list, 通常 1 个 768-D)
2. Line 573-586: 内部 if 检测 `use_structural_routing` → 运行 `structural_router` → 逐个 token append 到 `gcn_feats`
3. 最终 `gcn_feats` = [gcn_pooled] + [str_tok_1, ..., str_tok_6] = 7 个 768-D features
4. Line 600-606: `equal_concat` → L2-norm 每个 feat → concat → test_feat = global_norm + 7*part_norm = 8*768 = 6144-D

**通过。** 测试特征包含 global + GCN + 6 个 STD-PR tokens, 符合设计意图。

---

## PLTD (Part-Level Token Dropout) 在 dual branch

**状态: 未在 dual branch 中应用** (v2 M4 仍存在)。

Line 374-392 的 PLTD 逻辑只在 STD-PR standalone 路径中。dual branch 的 STD-PR block (line 438-462) 没有 PLTD。

影响: 如果 `POSE_STR_PART_DROP > 0`, standalone 和 dual 模式训练行为不一致。但 exp205 config 可能不启用 PLTD, 此时无影响。

**严重程度: Medium** — 不阻塞训练, 但如果未来需要 PLTD + dual, 需注意。

---

## 向后兼容性验证

| 模式 | 路径 | 影响 |
|------|------|------|
| STD-PR standalone (structural=True, gcn=False) | Train: line 356 if 条件 True → STD-PR path | 无影响, `not self.use_skeleton_gcn` = True |
| GCN standalone (structural=False, gcn=True) | Train: line 356 False → line 433 True, dual_branch_active=False | 无影响, line 438 `getattr('use_structural_routing')` = False |
| 无 part branch | Train: line 519 scalar return | 无影响 |
| PSG only (no gcn, no str) | 同上 | 无影响 |

**通过。** 所有已有实验的行为不变。

---

## 结论

**审查通过**

v2 的两个 Critical 问题 (训练/测试路径 if/elif 分支不可达) 已正确修复。`not self.use_skeleton_gcn` guard 精确地排除了 dual 场景从第一个 `if` 进入, 使流程正确落入 `elif` 块的 dual branch 代码。

### 遗留问题 (不阻塞训练)

1. **Medium**: Additive CE 和 Triplet 的 `score[1:]` / `feat[1:]` 包含 GCN token (降级自 v2 H1)。不会导致错误, 但 GCN 的 loss 权重与 STD-PR tokens 混合平均。
2. **Medium**: PLTD 未在 dual branch 的 STD-PR block 中应用 (v2 M4)。
3. **Low**: `_pg` 列表定义三次 (v2 L3)。
