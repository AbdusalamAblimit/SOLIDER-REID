# exp217 Claude Review — OERL (Occlusion-Equivariant Representation Learning)

## 审查范围

- `experiments/exp217/design.md`
- `processor/processor.py` (OERL block, lines 870-925)
- `config/defaults.py` (OERL defaults)
- `model/pose_backbone_model.py` (forward return format, feat_maps structure)
- `datasets/pose_dataset.py` (OA-SD mode, PLBOA occlusion handling)
- `datasets/make_dataloader.py` (OA-SD activation)

---

## 发现的问题

### Issue 1 — Critical: 可见性判断不正确

**位置**: `processor/processor.py` line 911

```python
visible = kp_w > 0.3  # (B, 17) — visible in occluded view
```

`kp_w` 来自 `kp_data['kp_weights']`，由 GCN head 根据 `pose_dict` 中的原始 keypoint scores 计算。PLBOA（`_apply_lower_body_occlusion`）**只修改图像像素，不修改 `persons` 数据的 scores**。因此 `kp_w > 0.3` 对所有原始可见 keypoints 都为 True，包括被 PLBOA 遮挡的那些。

**影响**: POI loss 会尝试对齐被遮挡 keypoints 处的特征（clean vs occluded），完全违背了"仅对齐可见部分"的核心设计意图。训练集 95.8% 可见意味着几乎所有 17 个 keypoints 都被标记为 visible，POI 退化为全局特征对齐。

**修复方案**: 需要在 PLBOA 之后更新 pose_dict 中的 visibility/scores，或在 OERL 中根据 PLBOA 的遮挡区域独立计算哪些 keypoints 被遮挡。例如，保存 PLBOA 的 `occ_start` 坐标，将 y > occ_start 的 keypoints 标记为 occluded。

### Issue 2 — High: OERL 对 POSE_OA_SD 有隐性依赖

**位置**: `processor/processor.py` line 872

```python
if oerl_enabled and (oa_sd_mode or parallel_oa_sd) and use_pose and img_teacher is not None:
```

OERL 需要 `img_teacher`（clean pre-PLBOA image），这只在 `POSE_OA_SD=True`（或 `POSE_OA_RD=True`）时由 dataset 提供。如果只启用 `POSE_OERL=True` 而不启用 OA-SD，OERL 会**静默跳过**，不报错也不产生任何效果。

**影响**: 用户可能启用 OERL 但忘记启用 OA-SD，训练正常运行但 OERL 完全无效，浪费实验时间。

**修复方案**: 在 `make_dataloader.py` 中检测：如果 `POSE_OERL=True` 但 `POSE_OA_SD=False` 且 `POSE_OA_RD=False`，抛出明确错误或自动启用 `_oa_sd_mode`。

### Issue 3 — High: parallel_aug 路径下 feat_maps 未定义

**位置**: `processor/processor.py` lines 473-492 vs line 890

在 `parallel_aug and use_pose` 分支中，model output 的 feature maps 被保存为局部变量 `fm_v`，但 `feat_maps` 从未被赋值。当 OERL 在 line 890 检查 `if feat_maps is not None` 时，会触发 `NameError`。

**触发条件**: `POSE_PARALLEL_AUG + POSE_OA_SD + POSE_OERL` 同时启用。

**当前影响**: 如果实验只用 2-view 模式（`POSE_OA_SD=True, POSE_PARALLEL_AUG=False`），不会触发此 bug，因为 `elif use_pose` 分支正确设置了 `feat_maps`。但属于潜在崩溃风险。

**修复方案**: 在 parallel_aug 路径中也设置 `feat_maps = fm_v`（或从 view 0 获取）。

### Issue 4 — Medium: clean forward 执行了不必要的 GCN/STD-PR 分支

**位置**: `processor/processor.py` line 876

```python
clean_out = model(img_teacher, label=target, ...)
```

clean forward 调用完整 model forward，包括 GCN head、STD-PR routing 等所有分支。这些分支的输出（cls_scores、part features）全部被丢弃（line 882: `_, clean_feat, ...`）。

**影响**: 
- 额外的 GPU 显存消耗（GCN head 的中间激活值）
- 额外的计算时间
- GCN 使用 `featmaps[-1].detach()`，所以 GCN 分支不影响 backbone 梯度（无害但浪费）

**修复方案**: 考虑为 model 添加 `forward_backbone_only=True` 模式，或直接调用 `model._run_backbone_with_psg()` 仅获取 feature maps。

### Issue 5 — Medium: design.md 与实现不一致

**设计文档描述**:
- 5 种遮挡模式（左半身、右半身、上半身、下半身、随机 40-60%）
- 实现 `random_pose_occlusion()` 函数

**实际实现**:
- 完全依赖 PLBOA 的遮挡（主要是下半身遮挡）
- 没有实现 `random_pose_occlusion()` 函数
- 遮挡模式单一，不是设计中描述的多样化遮挡

**影响**: 实验的遮挡多样性远低于设计预期。如果 POI loss 只在下半身遮挡场景下训练，泛化能力受限。

### Issue 6 — Low: 显存估计过于乐观

设计文档称"6GB * 2 = 12GB 可行"。实际上：
- 单次 forward (Swin-Tiny, bs=64, 384x128) 含激活值约 8-10GB
- 两次 WITH-gradient forward 需要同时保留两个计算图的激活值
- 加上 GCN head、optimizer states 等，总显存可能达 16-20GB
- 3090 (24GB) 可能仍可行，但余量不大

**建议**: 首个 epoch 密切监控 GPU 显存。如果 OOM，可考虑 gradient checkpointing (`WITH_CP=True`) 或 clean forward 仅对 backbone 部分保留梯度。

### Issue 7 — Low: OERL_OCC_RATIO 配置未被使用

`config/defaults.py` 中定义了 `POSE_OERL_OCC_RATIO = 0.5`，但 processor 中的 OERL 实现完全依赖 PLBOA 的现有遮挡，没有使用此参数。

---

## 正确性确认

以下方面审查通过：

1. **clean forward 无 torch.no_grad()**: 确认。line 876 直接调用 model forward，梯度正常流动。
2. **feat_maps[-1] 非 detached**: 确认。`_run_backbone_with_psg()` 返回的 `outs` 中的 feature maps 保留梯度。
3. **clean_out[2][-1] 访问正确**: 确认。model forward 返回的 position [2] 始终是 `featmaps`（一个 list），`[-1]` 取最后一个 stage 的 feature map。
4. **grid_sample 坐标映射**: 基本正确。`kp/input_size * 2 - 1` 配合 `align_corners=True` 合理。
5. **AMP 兼容性**: `grid_sample` 和 `F.normalize` 在 AMP 下安全。
6. **无 double CE loss**: 确认。clean forward 的 cls_scores 未参与任何 loss 计算。
7. **_loss_details 累积模式**: 遵循项目既有模式，正确。
8. **defaults.py 安全**: OERL 默认关闭 (`False`)，不影响现有实验。

---

## 审查结论

**未通过审查。** 必须修复 Issue 1 (Critical) 和 Issue 2 (High) 后才能启动训练。Issue 3 取决于实验配置，如果确认只用 2-view 模式可暂缓，但建议一并修复。

### 修复优先级

1. **Issue 1 (Critical)**: 修复可见性判断 — 这是核心创新点的根基
2. **Issue 2 (High)**: 添加 OERL-OA_SD 依赖检查
3. **Issue 3 (High)**: 修复 parallel_aug 路径的 feat_maps 未定义
4. **Issue 5 (Medium)**: 更新 design.md 或实现多样化遮挡
5. **Issue 4, 6, 7 (Medium/Low)**: 可在功能验证后优化
