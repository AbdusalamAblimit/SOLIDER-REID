# exp209 Review: Small + STD-PR + CE + OA-SD (远程 1-view)

## 审查范围

- design.md 合理性与创新门槛
- SupCon 关闭后 loss 路径回退到 per-token CE 的正确性
- OA-SD 与 STD-PR per-token list feat 的兼容性 (zip 对齐)
- dataset 2-view 生成 (OA-SD mode, 非 parallel_aug)
- EMA teacher 创建、forward、更新
- PLBOA 启用验证 (OA-SD 前提)
- 内存估算 (远程 16GB)
- config override 正确性

## 1. 创新门槛评估

本实验是 recipe 交叉验证：OA-SD 已在 GCN+CE (exp191) 上验证有效 (+2.9/+2.6)，STD-PR+CE 已在 exp166 中验证。这里组合两者看 OA-SD 是否与架构无关。作为 ablation 数据点可接受，非主线创新。

## 2. SupCon 关闭后 loss 路径 — 正确回退到 per-token CE

`make_loss.py:160`: `elif getattr(cfg.MODEL, 'POSE_STR_SUPCON', False) and ...` — 当 `POSE_STR_SUPCON=False` 时，此条件为 False。

代码走到 `else` 分支 (line 202-204):
```python
part_ids = [ce_fn(s, target) for s in score[1:]]
part_id_avg = sum(part_ids) / len(part_ids)
```
即标准 per-token CE。`score[1:]` 包含 6 个 per-token logits，逐个算 CE 取平均。

`make_loss.py:206`: `if getattr(cfg.MODEL, 'POSE_STR_SUPCON', False) and ...` — global SupCon 也不会执行。

**Triplet 路径** (line 244-248): 无 MaxSim triplet (默认 False)，走标准 per-token triplet:
```python
use_norm = len(feat) > 3  # True (7 > 3)
part_tris = [triplet(f, target, normalize_feature=True)[0] for f in feat[1:]]
```
6 个 per-token triplet，normalize_feature=True。正确。

**结论**: 无 bug，CE + triplet 完整覆盖 global + 6 per-token。

## 3. OA-SD 与 STD-PR per-token feat 兼容性 — 完全兼容

Model forward (pose_backbone_model.py:425) 在 STD-PR per-token 模式返回:
```
feat = [global_feat] + str_feat_list  # len=7 (global + 6 tokens)
```

Student 和 teacher 使用相同 model architecture，所以:
- `feat` = list of 7 tensors (student)
- `teacher_feat` = list of 7 tensors (teacher, via EMA deepcopy)

processor.py:713-721:
```python
elif isinstance(feat, list) and isinstance(teacher_feat, list):
    for sf, tf in zip(feat, teacher_feat):
        ...cosine distillation...
    oa_sd_loss = sum(distill_losses) / len(distill_losses)
```
`zip` 逐元素对齐：global↔global, tok1↔tok1, ..., tok6↔tok6。7 对 distillation 取平均。

`POSE_OA_SD_GLOBAL_ONLY` 默认 False — 这里无 SupCon 冲突，distill 所有 tokens 是正确选择（CE 梯度与 OA-SD 梯度不冲突）。

**结论**: 兼容，无 shape mismatch。

## 4. Dataset 2-view 生成 — 正确

`make_dataloader.py:126-127`: `POSE_OA_SD=True` → `train_set._oa_sd_mode = True`。

`pose_dataset.py:171-172`: OA-SD mode 下，PLBOA 前保存 `img_clean_for_oa_sd = img.copy()`。

`pose_dataset.py:263-269` (标准 1-view 路径): student (post-PLBOA + RE) 与 teacher (pre-PLBOA, mild RE) 组合为 `(img_tensor, img_clean_tensor)` 二元组。

`processor.py:442`: `len(img)==2` → `oa_sd_mode=True`。分离 student/teacher views。

不使用 `POSE_PARALLEL_AUG` (3-view)，只有 2-view OA-SD。设计文档说 "1-view" 应理解为无 parallel_aug 的 3-view，实际是 2-view (student + teacher)。预期行为。

**结论**: 数据流正确。

## 5. PLBOA 启用验证 — 已启用

Config (line 29-30): `POSE_LOWER_BODY_OCC: True`, `POSE_LOWER_BODY_OCC_PROB: 0.7`。

这是 OA-SD 的核心前提。如果关闭，teacher 和 student 看几乎相同的图，distillation 退化。

**结论**: OK。

## 6. 内存估算 — 应设 WITH_CP=True

Swin-Small: ~50M params。EMA teacher (deepcopy, no grad): ~200MB。
Student forward+backward (bs=64, 384x128, 18-block Stage 3): ~8-10GB。
Teacher forward (no_grad): ~2-3GB。
总计: ~11-14GB → 16GB 可行但偏紧。

exp206 (同为 Small + OA-SD) 的 review 建议 `WITH_CP: True`。保持一致。

**需确认**: 命令行是否包含 `MODEL.WITH_CP True`？如果基础 config 中 `WITH_CP: False` (line 14)，需要显式 override。

## 7. Config Override 检查

基础 config: `pose_psg_stdpr_pertoken_plboa_pape_ms_supcon_small.yml`

需要的命令行 override:
- `MODEL.POSE_STR_SUPCON False` — 关闭 SupCon ✓
- `MODEL.POSE_OA_SD True` — 启用 OA-SD ✓
- `MODEL.WITH_CP True` — 内存安全 (建议)
- `OUTPUT_DIR ./log/occluded_duke/exp209_...` — 独立输出目录

不需要改的 (已在基础 config 中):
- `POSE_STRUCTURAL_ROUTING: True` ✓
- `POSE_STR_PER_TOKEN: True` ✓
- `POSE_LOWER_BODY_OCC: True` ✓
- `POSE_PATCH_EMBED: True` ✓
- `POSE_PSG_STAGES: [2, 3]` ✓
- `PRETRAIN_PATH: pretrained/swin_small.pth` ✓
- `BASE_LR: 0.0004` ✓

无需 `POSE_OA_SD_GLOBAL_ONLY` (保持默认 False, CE 路线无冲突)。
无需 `POSE_PARALLEL_AUG` (不用 3-view)。

## 8. 潜在风险点

**无 Critical 或 High 级问题。**

- **Medium**: `WITH_CP` 需确认是否在命令行中设置。base config 为 False，16GB 可能偏紧。
- **Low**: SupCon 温度参数 `POSE_STR_SUPCON_TEMP: 0.05` 在 config 中仍存在但无效 (SUPCON=False 时不读取)。无害。

## 审查结论

**审查通过**。SupCon 关闭后正确回退到 per-token CE，OA-SD 与 STD-PR per-token 输出完全兼容 (7 元素 list zip 对齐)，dataset 2-view 生成正确，PLBOA 已启用。唯一建议是确保 `WITH_CP True` 在命令行中以避免 16GB OOM。无需新代码。
