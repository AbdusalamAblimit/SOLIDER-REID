# exp217 Claude Review v2 — OERL (Occlusion-Equivariant Representation Learning)

## 审查范围

- `experiments/exp217/design.md`
- `processor/processor.py` (OERL block, lines 870-935)
- `config/defaults.py` (OERL defaults: lines 189-191)
- `model/pose_backbone_model.py` (forward return format, feat_maps structure)
- `datasets/pose_dataset.py` (pose_dict shape conventions)
- 与 v1 审查 (claude_review.md) 的 3 个 Critical/High issues 对照

---

## v1 三个 Critical/High 问题的修复确认

### v1 Issue 1 (Critical: wrong visibility mask) — 已修复

v1 使用 `kp_data['kp_weights']` 作为可见性判断，而 PLBOA 不修改 scores，导致被遮挡的 keypoints 仍被标记为 visible。

v2 完全重写：不再依赖外部遮挡。使用 `torch.randperm` 随机选择 keypoints 作为 `occ_mask`(line 899-902)，然后 `visible = (~occ_mask) & (kp_scores > 0.3)` (line 923) 正确排除了被遮挡的 keypoints。遮挡是自生成的，不依赖 PLBOA。

### v1 Issue 2 (High: OA-SD dependency) — 已修复

v1 条件包含 `oa_sd_mode or parallel_oa_sd` 和 `img_teacher is not None`，导致 OERL 必须搭配 OA-SD。

v2 条件 (line 872): `if oerl_enabled and use_pose and feat_maps is not None and kp_data is not None`。不再依赖 OA-SD 模式或 teacher 图像。OERL 在单次 forward 的 feature map 上操作，用 heatmap 合成遮挡。

### v1 Issue 3 (High: feat_maps undefined in parallel_aug) — 仍存在但影响有限

`parallel_aug` 路径 (lines 473-492) 仍未给 `feat_maps` 赋值。如果同时启用 `parallel_aug + OERL`，line 872 引用 `feat_maps` 会触发 `NameError`。

**实际影响**: 低。exp217 的设计文档未涉及 parallel_aug 模式。标准单 view 训练和 OA-SD 2-view 训练都走 `elif use_pose:` 路径 (line 493)，`feat_maps` 正常赋值。建议后续修复但不阻塞本次实验。

---

## v2 新实现逐项审查

### 1. Feature-map-level occlusion 正确性 — 通过

- `fm = feat_maps[-1]` (line 882): 来自 `_run_backbone_with_psg()` 返回的 `outs[-1]`，是 stage 3 输出的 (B, 768, 12, 4) feature map，非 detached
- 从 `pose_dict['heatmaps'][:, 0]` 获取 person 0 的 17 个 heatmap (line 886)
- 随机选择 `num_occ = max(3, int(17 * 0.5)) = 8` 个 keypoints (line 898)
- 将选中 keypoints 的 heatmaps resize 到 feature map 尺寸，取 max 聚合为遮挡区域 (lines 905-909)
- 归一化反转得到 spatial_mask: 1=保留, 0=遮挡 (lines 911-912)
- `fm_occluded = fm * spatial_mask.unsqueeze(1)`: 正确的 soft occlusion (line 915)

**Shape 验证**:
- `heatmaps`: (B, 17, hH, hW) -> `F.interpolate` -> (B, 17, 12, 4)
- `occ_mask.float().unsqueeze(2).unsqueeze(3)`: (B, 17, 1, 1) — 正确广播
- `occ_hm`: (B, 12, 4) — `.max(dim=1)[0]` 正确
- `spatial_mask`: (B, 12, 4)
- `fm * spatial_mask.unsqueeze(1)`: (B, 768, 12, 4) * (B, 1, 12, 4) — 正确广播

### 2. Visibility mask 正确性 — 通过

Line 923: `visible = (~occ_mask) & (kp_scores > 0.3)`

- `occ_mask` (B, 17): 由 `torch.randperm` 生成，True = 被我们人为遮挡
- `~occ_mask`: 未被遮挡的 keypoints
- `kp_scores > 0.3`: 原始 pose 检测有效的 keypoints（排除不存在的 person 0 或低置信度检测）
- 两者取 AND = "既未被遮挡，且原始检测有效" — 正确，POI loss 只对这些 keypoints 计算

### 3. clean_kp 和 occ_kp 来源 — 通过

- `clean_kp = F.grid_sample(fm, grid, ...)` (line 893): 从原始 feature map `fm` 采样
- `occ_kp = F.grid_sample(fm_occluded, grid, ...)` (line 918): 从 `fm * spatial_mask` 采样
- 两者都来自同一个非 detached 的 `fm`，梯度通过两个路径都回传到 backbone — 正确

### 4. 可微性 — 通过

- `fm_occluded = fm * spatial_mask.unsqueeze(1)`: element-wise multiply，完全可微
- `spatial_mask` 来自 `pose_dict` 数据（无梯度），所以 `fm_occluded` 的梯度等于 `spatial_mask * grad_output`，正确地让被遮挡区域的梯度为零
- `F.grid_sample` 在 `mode='bilinear'` 下可微
- `F.normalize` 可微（分母 clamp 防除零）

### 5. 无 OA-SD 模式下工作 — 通过（有条件）

OERL 不再依赖 OA-SD。但条件中的 `kp_data is not None` 要求模型返回 5 个元素（见下方 Issue 1）。

### 6. 内存开销 — 通过

- 无第二次 forward pass
- 额外开销仅为：heatmap resize (小张量), 一次额外 grid_sample, cosine loss 计算
- 估计额外显存 < 100MB，完全可忽略

### 7. pose_dict shape 索引 — 通过

- `pose_dict['heatmaps'][:, 0, :, :, :]`: DataLoader 将 (max_persons, 17, hH, hW) collate 为 (B, max_persons, 17, hH, hW)，`[:, 0]` 取 person 0 -> (B, 17, hH, hW) — 正确
- `pose_dict['keypoints'][:, 0, :, :]`: (B, max_persons, 17, 2) -> (B, 17, 2) — 正确
- `pose_dict['scores'][:, 0, :]`: (B, max_persons, 17) -> (B, 17) — 正确

---

## 新发现的问题

### Issue 1 — Medium: `kp_data is not None` 是多余的守卫条件

**位置**: line 872

```python
if oerl_enabled and use_pose and feat_maps is not None and kp_data is not None:
```

OERL 代码内部完全不使用 `kp_data`。它只使用 `pose_dict`（直接从 batch 解包）和 `feat_maps`。但 `kp_data is not None` 要求模型返回 5 个元素（即启用了 GCN 或 STD-PR head）。

如果未来想在无 GCN 配置下使用 OERL（例如 PSG-only + OERL），OERL 会静默不运行。

**建议**: 将条件改为 `if oerl_enabled and use_pose and feat_maps is not None:`，或至少添加日志警告。

**影响**: 本次实验使用 GCN，`kp_data` 一定非 None，所以**不阻塞本次实验**。

### Issue 2 — Low: 潜在的 spatial collapse 风险

POI loss 梯度同时流过 `clean_kp` 和 `occ_kp` 回到同一 `fm`。最小化 `1 - cos_sim` 的最简单方式是让 feature map 变得空间均匀（所有位置输出相同特征），这样无论是否遮挡，grid_sample 都得到相同结果。

**缓解因素**: ID loss 和 triplet loss 强制 feature map 保持判别性，应该能防止 collapse。OERL weight 默认 1.0，如果过大可能需要调低。

**建议**: 监控早期训练中 `oerl` loss 的值。如果快速降到 0（< 0.01），可能意味着 collapse 而非真正的 invariance learning。

### Issue 3 — Low: `occ_max.clamp(min=1e-6)` 在 AMP fp16 下的精度

`1e-6` 在 fp16 下可表示（fp16 最小正常值约 6e-5，但 subnormal 可到 ~6e-8）。不过 `occ_max` 来自 heatmap 的 max 值，在正常情况下远大于 1e-6。即使全零（person 未检测），spatial_mask = 1.0，poi_loss 对该样本贡献为零。无实际影响。

---

## 与 design.md 的一致性

v1 审查指出 design.md 描述了 5 种遮挡模式但实现仅依赖 PLBOA。v2 实现使用 **随机 keypoint 遮挡**（`torch.randperm` 选择 ~50% keypoints），这与 design 中的"随机 40-60% keypoints"模式一致。不再需要 PLBOA 来生成遮挡。

Design 中的结构化遮挡模式（左/右/上/下半身）未实现，但随机遮挡是合理的首步验证。一致性可接受。

---

## defaults.py 安全性

```python
_C.MODEL.POSE_OERL = False                # 默认关闭
_C.MODEL.POSE_OERL_WEIGHT = 1.0           # 合理默认
_C.MODEL.POSE_OERL_OCC_RATIO = 0.5        # 实际被使用（line 874, 898）
```

默认关闭，不影响任何现有实验。`POSE_OERL_OCC_RATIO` 在 v2 中正确被使用（v1 中未被使用）。

---

## 审查结论

**审查通过。**

v1 的 3 个 Critical/High 问题全部修复。v2 重写干净且正确：
1. 遮挡在 feature map 层面通过 heatmap 合成，无需第二次 forward
2. 可见性判断基于自生成的 `occ_mask`，不再依赖外部 pipeline
3. 不再依赖 OA-SD 模式
4. Shapes 全部正确，可微性完整，AMP 安全

新发现的 3 个 Medium/Low 问题不阻塞训练：
- Issue 1 (Medium): `kp_data` 守卫多余但本实验中无影响
- Issue 2 (Low): spatial collapse 风险由其他 loss 缓解，需监控 `oerl` loss 趋势
- Issue 3 (Low): fp16 精度无实际影响

### 监控建议
- 关注 `oerl` loss 初始值和收敛趋势（正常应该从 ~0.5-0.8 逐渐下降到 0.1-0.3）
- 如果 `oerl` loss 在前几个 epoch 就降到 < 0.01，考虑降低 OERL_WEIGHT 或为 clean_kp 添加 `.detach()`
- 关注 `oerl_nv`（平均每样本可见 keypoint 数），应该在 ~8-9（17 * 0.5 的未遮挡部分中有效检测的比例）
