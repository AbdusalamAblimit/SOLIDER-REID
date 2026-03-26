# exp193 Claude Review: OA-SD + 3-view Parallel Aug + CE

## 审查范围

1. `datasets/pose_dataset.py` — 4-view tuple 生成逻辑 (lines 167-226, 260-271)
2. `datasets/pose_dataset.py` — collate 函数 (lines 1054-1070)
3. `processor/processor.py` — multi-view 检测和分发 (lines 433-454)
4. `processor/processor.py` — parallel_aug loss 平均 + OA-SD loss 计算 (lines 459-656)
5. `processor/processor.py` — EMA teacher update (lines 678-683)
6. `datasets/make_dataloader.py` — flag 设置 (lines 122-127)
7. `experiments/exp193/design.md` — 实验设计
8. 与 standard OA-SD (2-view) 和 standard parallel_aug (3-view) 的交互

---

## 发现

### [Low] OA-SD 教师视图在 parallel_aug 模式下不做 Random Erasing

**位置**: `datasets/pose_dataset.py` lines 221-226 vs lines 264-268

在 standard OA-SD 模式下，教师视图会执行 mild RE:
```python
# Standard OA-SD (lines 264-268):
if img_clean_for_oa_sd is not None:
    img_clean_tensor = self._image_to_tensor(img_clean_for_oa_sd)
    if random.random() < self.re_prob:       # <-- RE applied
        img_clean_tensor, _ = self._random_erase(img_clean_tensor)
    img_tensor = (img_tensor, img_clean_tensor)
```

在 parallel_aug + OA-SD 模式下，教师视图不做 RE:
```python
# Parallel OA-SD (lines 221-224):
if img_clean_for_oa_sd is not None:
    img_clean_tensor = self._image_to_tensor(img_clean_for_oa_sd)
    # <-- NO RE applied
    img_tensor = (..., img_clean_tensor)
```

**分析**: 这不是 bug，反而可能是更好的设计选择。OA-SD 的核心是让教师看 clean image、学生看 occluded image，教师不做 RE 意味着更大的 student-teacher 视差，这与 OA-SD 的 motivation 一致。但与 standard OA-SD 行为不完全一致。如果需要严格消融对照，可能需要注意这一点。

**严重度**: Low — 不会导致错误，且语义上合理。

---

### [Medium] OA-SD loss 在 parallel_aug 平均之后添加 — 数值影响需确认

**位置**: `processor/processor.py` lines 606-656

Loss 计算顺序:
1. `loss = loss_fn(score, feat, ...)` — 仅 view 1 的 CE + triplet (line 532)
2. 加 recon_loss, LTCS, LPCS 等辅助 loss (lines 550-604)
3. **parallel_aug 平均** (lines 606-615): `loss = (loss + view2_loss + view3_loss) / 3`
4. **OA-SD loss 在平均之后添加** (lines 617-656): `loss = loss + oa_sd_weight * oa_sd_loss`

这意味着 OA-SD loss 不参与 `/3` 的平均，而是在平均后的 loss 之上直接加。这与 standard OA-SD 模式（直接 `loss + oa_sd_weight * oa_sd_loss`）的数值行为一致。

**分析**: 这实际上是正确的设计。OA-SD 只做一次 teacher forward + 一次 distillation，不需要参与 3-view 平均。但需要注意：辅助 loss (recon_loss, LTCS, LPCS) 如果存在，它们只针对 view 1 计算但也被除以 3（因为它们加到 loss 后再做平均）。这在 standard parallel_aug 中也是如此，不是新问题。

**严重度**: Medium — 不是 bug，但 view 1 的辅助 loss 被稀释到 1/3。如果 OA-SD weight 需要 tuning，需要意识到 base loss 已经被 /3 了。

---

### [通过] 数据流正确性: 4-view tuple 从 dataset 到 processor

1. **Dataset** (`pose_dataset.py` line 224): 返回 `(full, roa, heavy, clean)` — 4-element tuple
2. **Collate** (`pose_dataset.py` lines 1057-1065): `n_views = len(img_tuples[0])` = 4, 返回 `[tensor_v1, tensor_v2, tensor_v3, tensor_v4]` — list of 4 tensors, each (B,C,H,W)
3. **Processor** (`processor.py` line 427): `img, vid, target_cam, target_view, pose_dict = batch_data` — img is list of 4 tensors
4. **Detection** (`processor.py` lines 436-439):
   - `parallel_aug = isinstance(img, list) and len(img) >= 3` → True (4 >= 3)
   - `oa_sd_mode = isinstance(img, list) and len(img) == 2` → False (4 != 2)
   - `parallel_oa_sd = parallel_aug and oa_sd_enabled and len(img) == 4` → True

**结论**: 数据流完全正确。4-view tuple 正确地传递并被识别为 parallel_oa_sd 模式。

---

### [通过] 变量作用域: `img_teacher` 在 OA-SD 代码块执行时一定已定义

在 `parallel_oa_sd` 为 True 时 (line 441-443):
```python
if parallel_oa_sd:
    img_views = [v.to(device) for v in img[:3]]
    img_teacher = img[3].to(device)         # <-- defined here
```

OA-SD 代码块 (line 618):
```python
if oa_sd_enabled and (oa_sd_mode or parallel_oa_sd) and ...
```

`parallel_oa_sd` 为 True 时，`img_teacher` 一定在 line 443 被赋值。
`oa_sd_mode` 为 True 时，`img_teacher` 在 line 449 被赋值。

不存在 `img_teacher` 未定义的路径。

---

### [通过] 特征引用: distillation 使用 view 1 的 features

Line 477: `score, feat = all_scores[0], all_feats[0]`

OA-SD distillation 使用 `feat` (line 638-652)，即 view 1 (full view) 的特征。

**分析**: 合理。View 1 是"最接近正常"的视图（标准 RE），用它做 distillation target 与 standard OA-SD 一致——standard OA-SD 也是用 occluded student 的 features 对 clean teacher 做 distillation。这里 view 1 = post-PLBOA + maybe RE，教师 = pre-PLBOA clean，语义对齐。

---

### [通过] 标准模式不受影响

- **Standard OA-SD (2-view)**: `len(img) == 2` → `parallel_aug = False`, `oa_sd_mode = True`, `parallel_oa_sd = False`. 走 `elif oa_sd_mode` 分支 (lines 447-451)，完全不受新代码影响。
- **Standard parallel_aug (3-view)**: `len(img) == 3` → `parallel_aug = True`, `oa_sd_mode = False`, `parallel_oa_sd = False`（因为 `oa_sd_enabled` 为 False 或 `len(img) != 4`）。走 `if parallel_aug / if not parallel_oa_sd` 分支 (line 445)，完全不受影响。
- **Standard single-view**: `isinstance(img, Tensor)` → 无 list，走 `else` 分支 (lines 452-454)。

---

### [通过] Collate 函数处理 4-element tuples

`pose_train_collate_fn` (lines 1054-1070):
```python
n_views = len(img_tuples[0])  # = 4
if n_views == 1: ...
else:
    imgs = [torch.stack([t[v] for t in img_tuples], dim=0)
            for v in range(n_views)]  # 4 tensors, each (B,C,H,W)
```

这是完全通用的——不依赖于具体 view 数量，只要 `n_views > 1` 就走 else 分支。4-view 与 3-view 和 2-view 使用完全相同的路径。

---

### [通过] EMA update 不受影响

Line 679-683: EMA update 在 optimizer step 之后，使用 `base_model.parameters()` (student) 更新 `ema_teacher.parameters()`。这与是否是 parallel_oa_sd 模式无关——student 的 parameters 在 backward + step 后已经更新，EMA update 只看参数值。

---

### [通过] 测试时无影响

测试时 `self.is_train = False`，所以:
- `img_clean_for_oa_sd = None`（line 171 条件不满足）
- `self.parallel_aug` 不影响（line 194 条件不满足）
- 返回 `(img_tensor,)` 单元素 tuple (line 271)
- Processor 中 `parallel_aug = False`, `oa_sd_mode = False`

---

### [通过] Config 安全

实验通过 `POSE_PARALLEL_AUG=True` + `POSE_OA_SD=True` 组合激活。两个 flag 已存在于 defaults.py。`make_dataloader.py` 中:
- Line 123: `POSE_PARALLEL_AUG` → `train_set.parallel_aug = True`
- Line 126-127: `POSE_OA_SD` → `train_set._oa_sd_mode = True`

两者独立设置，组合效果由 dataset 和 processor 中的逻辑自动处理。

---

### [Low] 显存开销

4 次 forward pass: 3 student (with grad) + 1 teacher (no_grad)。
- 3 student views 顺序执行，每次只保留一份 activation（因为是 `for v_img in img_views` 循环）
- Teacher forward 在 `torch.no_grad()` 下，不保存 activation
- 峰值显存 ≈ 1 student forward (activation) + 3 views 的 loss 图 + teacher forward (no activation)
- 比 standard 3-view 只多一次 no_grad forward，增量很小

**严重度**: Low — 不太可能 OOM on 3090 24GB。

---

### [Low] Design.md 里的理解不完全准确

Design.md line 15-16 写道:
> 代码上无需修改：OA-SD 已支持 parallel_aug path

但实际上有代码修改（dataset 中的 4-view 打包、processor 中的 `parallel_oa_sd` 检测）。Design.md line 29 更准确地讨论了 "6 forward" 的担忧，说明作者后来意识到需要特殊处理。

**严重度**: Low — design.md 内部稍有矛盾，但不影响代码正确性。

---

## 总结

| 级别 | 数量 | 详情 |
|------|------|------|
| Critical | 0 | — |
| High | 0 | — |
| Medium | 1 | 辅助 loss (recon/LTCS/LPCS) 在 parallel_aug 中被 /3 稀释（已有行为，非新引入） |
| Low | 3 | 教师视图 RE 不一致（合理）；显存增量小；design.md 小矛盾 |

所有发现均为既有行为的特性说明或低风险注意事项，无 bug、无 runtime error 风险、无逻辑错误。数据流完整、变量作用域正确、标准模式不受影响、collate 通用、EMA 安全、测试时无影响。

## 结论

**审查通过**
