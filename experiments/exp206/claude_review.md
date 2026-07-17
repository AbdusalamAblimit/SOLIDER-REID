# exp206 Review: Swin-Small + GCN+PAA + CE + OA-SD (远程 1-view)

## 审查范围

- design.md 合理性与创新门槛
- OA-SD 与 GCN 输出兼容性
- dataset 2-view 生成（OA-SD mode）
- processor OA-SD distillation 与 GCN list feat 兼容性
- EMA teacher 创建与更新
- 内存估算（16GB 远程 GPU）
- config 选项正确性
- PLBOA 必要性（OA-SD 前提）

## 1. 创新门槛评估 — 本实验是组合验证，非独立创新

exp206 组合两个已验证有效的技术：GCN+PAA (70.8 CE on Small) + OA-SD (+2.9 on Tiny CE)。
这属于 **recipe scaling 验证**，不是新创新。作为论文 main table 的一行数据有价值（证明 OA-SD 在更强 backbone/architecture 上依然有效），但不作为主线创新。可接受。

## 2. OA-SD 与 GCN 输出兼容性 — 完全兼容

GCN forward (pose_backbone_model.py:517) 返回:
```
[cls_score] + gcn_cls_scores, [global_feat] + gcn_feats, featmaps, None, kp_data
```
即 `feat = [global_feat, gcn_pooled_feat]`，len=2。

OA-SD distillation (processor.py:713-721):
- `isinstance(feat, list) and isinstance(teacher_feat, list)` → True
- `zip(feat, teacher_feat)` → 逐元素 cosine distillation (global + gcn_pooled)
- 两个 list 长度都为 2，zip 对齐正确

**结论**: 兼容，无 bug。

## 3. Dataset 2-view 生成 — 正确

`make_dataloader.py:126-127`: 当 `POSE_OA_SD=True` 时设置 `train_set._oa_sd_mode = True`。

`pose_dataset.py:167-172`: `_oa_sd_mode` 为 True 时，在 PLBOA 前保存 `img_clean_for_oa_sd = img.copy()`。

`pose_dataset.py:263-269` (standard 1-view path): 将 student (post-PLBOA + ROA + RE) 和 teacher (pre-PLBOA, 有 mild RE) 组合为 `(img_tensor, img_clean_tensor)` 二元组。

`processor.py:442-456`: `len(img) == 2` → `oa_sd_mode = True`，正确分离 student/teacher views。

**注意**: 这里不使用 parallel_aug (3-view)，只用 2-view OA-SD 模式。设计文档说 "1-view" 应理解为 "没有 parallel_aug 的 3-view"，但实际是 2-view（student + teacher）。这是预期行为。

**结论**: 数据流正确。

## 4. PLBOA 必要性 — 关键前提

processor.py:410-411: 如果 `POSE_LOWER_BODY_OCC=False`，会发出 WARNING: "Teacher and student see near-identical images"。

OA-SD 的核心原理是 teacher 看 clean 图，student 看 occluded (PLBOA) 图。如果不启用 PLBOA，两者看到几乎相同的图（只有 ROA 差异），distillation 退化为自回归。

design.md 说 "基于 pose_psg_gcn_paa_roa.yml + ... + PLBOA"。`pose_psg_gcn_paa_roa.yml` 本身不含 PLBOA。必须在 config 或命令行显式添加:
```
POSE_LOWER_BODY_OCC: True
POSE_LOWER_BODY_OCC_PROB: 0.7
```

**结论**: 必须启用 PLBOA，否则 OA-SD 无效。

## 5. EMA Teacher 内存与计算 — 可行但偏紧

Swin-Small: depths=(2,2,18,2), embed_dims=96 → ~50M params
GCN+PAA: ~0.5M params
EMA teacher (deepcopy, no grad): ~50.5M params × 4 bytes = ~200MB

内存分解:
- Student model params + optimizer states: ~50.5M × (4+8) bytes ≈ 600MB
- EMA teacher params (no optimizer): ~200MB
- Student forward+backward activations (bs=64, 384x128): ~8-10GB (Small 18-block Stage 3)
- Teacher forward activations (no_grad, 无 backward 图): ~2-3GB
- 总计: ~11-14GB

16GB GPU 可行但偏紧。如果 OOM，需要 `WITH_CP: True`（gradient checkpointing 减少 student activations）。建议预设 `WITH_CP: True` 以保险。

**结论**: 应设 `WITH_CP: True`。

## 6. Config 检查清单

基于 `pose_psg_gcn_paa_roa.yml` 需要修改:
- `TRANSFORMER_TYPE: 'swin_small_patch4_window7_224'` (Small)
- `PRETRAIN_PATH: 'pretrained/swin_small.pth'`
- `BASE_LR: 0.0004` (Small 用更小 LR，per swin_small.yml 惯例)
- `WITH_CP: True` (内存安全)
- `POSE_OA_SD: True`
- `POSE_OA_SD_WEIGHT: 1.0` (默认值)
- `POSE_OA_SD_EMA_DECAY: 0.999` (默认值)
- `POSE_LOWER_BODY_OCC: True`
- `POSE_LOWER_BODY_OCC_PROB: 0.7`
- `CHECKPOINT_PERIOD: 20`
- `OUTPUT_DIR: './log/occluded_duke/exp206_small_gcn_paa_ce_oasd'`

不需要 SupCon 相关选项 (CE 路线)。不需要 `POSE_PARALLEL_AUG` (2-view 即可)。
不需要 `POSE_OA_SD_GLOBAL_ONLY` (CE 路线无 SupCon 冲突，保持 False distill 所有 tokens)。

## 7. Loss 路径验证

CE 路线 + list feat:
- processor.py 的 `_compute_loss_all_feats` 对 list feat 逐元素计算 CE+triplet
- 隐式 0.5x global loss (list 路径)
- OA-SD distillation 叠加在总 loss 上 (`loss = loss + oa_sd_weight * oa_sd_loss`)

无 SupCon → 无梯度冲突问题。CE + OA-SD 互补（exp191 已验证）。

## 8. ROA 与 OA-SD 交互

ROA (Random Occlusion Augmentation) 在 dataset 层面操作。OA-SD teacher 使用 pre-PLBOA 的 clean image。但 ROA 在标准 1-view 路径中是在 PLBOA 之后应用 (pose_dataset.py:230-243)。

因此 student 同时受 PLBOA + ROA 遮挡，teacher 不受 PLBOA 但... 等等，teacher 也不受 ROA 吗？

检查: `img_clean_for_oa_sd` 在 line 172 保存，在 PLBOA 之前。但 ROA 在 line 230-243 应用于 `img`（student），不影响 `img_clean_for_oa_sd`。所以 teacher 是完全 clean 的（无 PLBOA，无 ROA）。这是正确行为——teacher 看最干净的图。

**结论**: ROA + OA-SD 交互正确。

## 审查结论

**审查通过**。exp206 是已验证组件 (GCN+PAA + OA-SD) 在 Swin-Small 上的 scaling 验证，代码层面完全兼容，无 runtime error 风险。

关键提醒:
1. **必须启用 PLBOA** (`POSE_LOWER_BODY_OCC: True`)，否则 OA-SD 退化
2. **建议 WITH_CP: True** 避免 16GB OOM
3. **BASE_LR 用 0.0004**（Small 惯例）
4. CHECKPOINT_PERIOD 设为 20 以便中间测试
