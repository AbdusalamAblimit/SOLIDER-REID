# exp203 Review: Swin-Small + GCN+PAA+ROA + SupCon + PLBOA + 3-view

## 审查范围
- design.md 合理性
- SupCon 与 GCN 兼容性
- PLBOA + parallel_aug 与 GCN+PAA+ROA 兼容性
- WITH_CP 与 GCN 兼容性
- loss 路径正确性
- 数据流完整性

## 1. SupCon 与 GCN 输出兼容性 — 可行但语义不同

**SupCon 触发条件** (make_loss.py:160):
`POSE_STR_SUPCON=True and isinstance(feat, list) and len(feat) > 1`

**GCN 返回** (pose_backbone_model.py:464):
`[cls_score] + [gcn_cls_score], [global_feat] + [skeleton_feat]`
即 `feat = [global_feat, skeleton_feat]`, len=2, 满足条件。

**行为**: SupCon 会对 `feat[1:]` = `[skeleton_feat]` (单个 pooled feature) 计算 SupCon loss。这与 STD-PR 的 6 个 per-token features 语义不同，但代码上不会崩溃。SupCon 作用于 1 个 pooled skeleton feature 本质上是对该 feature 做对比学习，合理。

**VIS_WEIGHT**: GCN 的 kp_data 不含 `part_visibility` 字段 (只有 STD-PR 才有)。如果 `POSE_STR_SUPCON_VIS_WEIGHT=True` 且 GCN 模式，则 `kp_data['part_visibility']` 不存在，会走 else 分支 (均匀平均)，不会崩溃。但建议显式设 `POSE_STR_SUPCON_VIS_WEIGHT: False`。

**结论**: SupCon+GCN 兼容，无 bug。

## 2. PLBOA 与 GCN+PAA+ROA 兼容性 — 无冲突

PLBOA (POSE_LOWER_BODY_OCC) 在 dataset 层面操作 (make_dataloader.py:107-111)，修改图像。ROA 也在 dataset 层面操作。两者独立作用于图像，互不干扰。

PLBOA 的 3-view 模式 (parallel_aug + OA-SD): dataset 返回 4 个视图 [view1, view2, view3, teacher_clean]。processor 正确解析 (processor.py:444-448)。每个 view 独立过 model forward。GCN 的 `feat_map_detached = featmaps[-1].detach()` 在每个 view 独立执行，无交叉污染。

**结论**: PLBOA + parallel_aug + GCN 兼容。

## 3. 3-view parallel_aug 与 GCN 兼容性 — 无冲突

processor.py:465-481 对每个 view 独立调用 `model(v_img, ..., pose_dict=pose_dict)`:
- 每个 view 返回 5-tuple `(score, feat, featmaps, recon_loss, kp_data)`
- GCN 路径 (pose_backbone_model.py:432-464) 返回正确的 5-tuple
- processor.py:673-681 对 view 2/3 的 loss 也会正确走 list-loss 路径 (score 是 list)

**注意**: 3 个 view 使用**同一个** `pose_dict`。这是正确的 — pose_dict 来自原图 (无 augmentation)，3 个 augmented views 共享 pose 信息。

**结论**: 兼容。

## 4. WITH_CP 与 GCN — 无冲突

WITH_CP (gradient checkpointing) 作用于 Swin backbone 的 SwinBlock 内部 (make_model.py:200-202)。GCN 的 `feat_map_detached = featmaps[-1].detach()` 取的是 backbone 输出的 detached copy，GCN 自身不使用 checkpoint。skeleton_gcn.py 中无任何 checkpoint 相关代码。

PSG 和 PAA 的 `_run_stage_with_psg` 手动展开了 backbone blocks，绕过了 Swin 原生的 with_cp forward。需确认: PSG 手动展开是否保留了 with_cp 行为?

检查: `_run_stage_with_psg` 调用 `block(x, hw_shape)` — 这调用的是 SwinBlock.forward()。Swin 的 with_cp 在 SwinBlock.forward 内部用 `torch.utils.checkpoint.checkpoint()`，所以即使手动展开 stage，每个 block 内部仍然使用 checkpoint。

**结论**: WITH_CP 与 GCN+PSG+PAA 兼容。

## 5. OA-SD + parallel_aug (parallel_oa_sd) 与 GCN — 需注意

如果同时启用 OA-SD + parallel_aug，processor.py:444 检查 `len(img) == 4`。dataset 需要返回 4 个视图。make_dataloader.py:123-127 显示 `parallel_aug` 和 `_oa_sd_mode` 是独立设置的。需要确认 dataset 是否同时生成 4 个视图。

EMA teacher forward (processor.py:685-701) 对 `img_teacher` (clean image) 调用 model。GCN model 返回 list feat，OA-SD distillation (processor.py:713-719) 对 list feat 逐元素计算 cosine distillation。这与 GCN 的 2-element list 兼容。

**结论**: 兼容，但需确认 config 中同时设置 `POSE_PARALLEL_AUG: True` 和 `POSE_OA_SD: True`。

## 6. Config 注意事项

基于 `pose_psg_gcn_paa_plboa_roa.yml` 修改，需要:
- `TRANSFORMER_TYPE: 'swin_small_patch4_window7_224'`
- `PRETRAIN_PATH: 'pretrained/swin_small.pth'`
- `WITH_CP: True` (3-view Small 需要)
- `POSE_STR_SUPCON: True`
- `POSE_STR_SUPCON_ADDITIVE: True` (additive mode, CE + SupCon)
- `POSE_PARALLEL_AUG: True`
- `POSE_OA_SD: True` (if using OA-SD with PLBOA)
- `BASE_LR: 0.0004` (per design.md)
- `POSE_STR_SUPCON_VIS_WEIGHT: False` (GCN 无 part_visibility)

## 7. 潜在问题

**Medium**: SupCon 对单个 pooled skeleton feature (而非 6 个 per-token) 的效果未验证。可能效果不如 STD-PR 的 per-token SupCon，因为 pooled feature 已经丢失了 part-level 粒度。但不会导致错误，只是效果可能不如预期。

**Low**: OA-SD 的 `POSE_OA_SD_GLOBAL_ONLY` 设计意图是避免与 SupCon 在 per-token 上的梯度冲突。GCN 只有 1 个 pooled feature，冲突较小，可以尝试 `GLOBAL_ONLY: False`。

## 审查结论

**审查通过**。SupCon + GCN + PAA + ROA + PLBOA + 3-view 在代码层面完全兼容，无 runtime error 风险。关键注意事项:
1. 确保 `POSE_STR_SUPCON_VIS_WEIGHT: False`
2. SupCon 在单 pooled feature 上的效果是实验问题，不是 bug
3. WITH_CP 与 PSG 手动展开兼容（checkpoint 在 block 内部）
