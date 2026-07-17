# exp235 FSDC 二次审查报告

审查人: Claude Opus 4.6
审查日期: 2026-04-03
审查范围: design.md, feature_denoiser.py (完整), pose_backbone_model.py (训练+测试分支), processor.py (FSDC loss), config/defaults.py, solver/make_optimizer.py

---

## 一次审查 Critical Issues 修复验证

### C1: 残差连接使用 `masked_tokens` — 已修复

**feature_denoiser.py:176**: `output = masked_tokens + self.output_proj(decoded)`

验证: 训练路径中, `masked_tokens` 在 masked 位置是 `mask_token + noise`, 在 visible 位置保持原始特征。`output_proj` 零初始化, 所以初始输出为 `masked_tokens` 本身。随着训练, denoiser 学习将 masked 位置从 `mask_token + noise` 修正回 `original_feature`。重建 loss = `MSE(mask_token + noise + correction, original)`, 初始 loss 非零, 梯度方向正确。**修复正确。**

### C2: 测试时 FSDC 调用 — 已修复

**pose_backbone_model.py:649-656**: 在 `elif self.use_skeleton_gcn` 测试分支中, FSDC 在 skeleton_head 之前被调用:
```python
feat_for_gcn = featmaps[-1]
if getattr(self, 'use_fsdc', False) and scene_heatmaps is not None:
    ...
    completed, _, _ = self.feature_denoiser(spatial_tokens, scene_heatmaps, ...)
    feat_for_gcn = completed.transpose(1, 2).reshape(...)
```
然后 `feat_for_gcn` 传入 `self.skeleton_head(feat_for_gcn, ...)` (line 658)。**修复正确。**

### C3: 测试时残差使用 `masked_tokens` — 已修复

**feature_denoiser.py:224**: `output = masked_tokens + self.output_proj(decoded)`

与训练路径一致, 使用 `masked_tokens` 而非原始 `spatial_tokens`。在测试时 occluded 位置使用 `mask_token` (无 noise), 经过 denoiser 修正后输出 completed features。**修复正确。**

### H3: 全可见 mask fallback — 已修复

**feature_denoiser.py:144-149**: 训练路径中, 在 body-part mask 生成后逐 sample 检查:
```python
for b in range(B):
    if mask[b].all():
        num_masked = max(1, int(N * self.mask_ratio))
        indices = torch.randperm(N, device=device)[:num_masked]
        mask[b, indices] = False
```
确保每个 sample 至少有 1 个 masked token, 不浪费训练信号。**修复正确。**

---

## 完整审查: feature_denoiser.py

### 训练路径 (forward, self.training=True)

1. **Mask 生成** (line 142-151): 有 heatmap 时用 body-part mask + per-sample fallback; 无 heatmap 时 random mask。逻辑正确。
2. **Target 保存** (line 154): `target = spatial_tokens.detach().clone()` — 正确, detached 且独立副本。
3. **Token 替换** (line 157-164): `torch.where(mask, keep, mask_token + noise)` — 逻辑正确, mask=True 保留, mask=False 替换。
4. **Positional + mask embedding** (line 167-168): `mask_indicator = mask.long()`, 1=visible, 0=masked。`mask_embed(0)` = masked embedding, `mask_embed(1)` = visible embedding。正确。
5. **Decoder** (line 171): `self.decoder(input_tokens, input_tokens)` — tgt 和 memory 相同。如 v1 审查 H1 所述, cross-attention 退化为第二次 self-attention。功能上无 bug, 但计算冗余。**保留 H1 为 acknowledged non-blocking**。
6. **残差 + 输出** (line 176): `masked_tokens + output_proj(decoded)` — 已修复, 正确。
7. **重建 loss** (line 179-186): 只在 masked positions 计算 MSE。`masked_positions = ~mask`。逻辑正确。当无 masked positions 时返回 `torch.tensor(0.0)` — 不会因 H3 修复而触发 (至少 1 个 masked token)。
8. **Stats** (line 188-193): 计算 `num_masked` 和 `mask_ratio`, 用于日志。正确。
9. **返回** (line 195): `(output, recon_loss, stats)` — 与调用方匹配。

### 测试路径 (forward, self.training=False)

1. **Occlusion 检测** (line 199-206): 有 heatmap 时, `hm_max = hm.max(dim=1)[0]` 取各 keypoint 最大激活。`threshold = mean * 0.5` 偏激进 (v1 M4)。非阻塞, 可在实验中调。
2. **Masking** (line 211-219): 与训练路径类似但无 noise (测试时不加噪)。正确。
3. **Decoder + residual** (line 222-224): 与训练路径一致。正确。
4. **No occlusion path** (line 226): 直接返回原始 `spatial_tokens`。正确, 无不必要计算。
5. **返回** (line 228): `(output, None, {})` — 与调用方匹配。

### _generate_body_part_mask (line 69-114)

1. **Part groups** (line 86-93): 6 组 COCO keypoints, 覆盖全部 17 个 keypoints。正确。
2. **Resize** (line 96): `F.interpolate(heatmaps, (fH, fW))` — AMP 下可能以 float16 执行 (v1 M3), 非阻塞。
3. **Random part selection** (line 102-103): `torch.randperm(6)[:num_parts_to_mask]` — 在 CPU 上执行 (未指定 device), 但 randperm 对小 N 无问题。
4. **Threshold** (line 110): `mean + 0.5 * std` — 合理的自适应 threshold。
5. **Mask update** (line 112): `mask[b] = mask[b] & ~part_mask.flatten()` — 逐步将 body part 区域标记为 masked (False)。正确。

### __init__ (line 30-67)

1. **mask_token** (line 39-40): 零初始化 + normal_(std=0.02)。标准 MAE 做法。
2. **pos_embed** (line 43-44): `trunc_normal_(std=0.02)` — 标准做法。
3. **mask_embed** (line 47): `nn.Embedding(2, feat_dim)` — 默认初始化 (normal)。可接受。
4. **TransformerDecoder** (line 50-56): `norm_first=True` (Pre-LN), `activation='gelu'`, `batch_first=True`。配置合理。
5. **output_proj** (line 59-61): 零初始化 weight 和 bias — 确保 identity start。正确。
6. **self.norm** (line 64): LayerNorm on decoded output, applied before output_proj input。正确。

---

## 完整审查: pose_backbone_model.py FSDC 集成

### 训练分支 (line 506-526)

1. **输入**: `feat_map_detached` (line 509) — 来自 line 471-475, 使用 GSPB gradient scaling。当 `_gs=0` 时完全 detach, 即 backbone 不受 FSDC 影响。正确。
2. **Flatten → denoiser → reshape** (line 510-514): Shape 变换正确: `(B, C, H, W) → (B, N, C) → denoiser → (B, N, C) → (B, C, H, W)`。
3. **传入 skeleton_head** (line 518-520): 使用 completed `feat_map_detached`。skeleton_head 的 CE/triplet loss 会反传到 denoiser (v1 H2)。这实际上是 end-to-end 训练的正确行为 — denoiser 同时接收重建 loss 和下游任务 loss, 两者方向一致。**H2 confirmed as desired behavior。**
4. **Loss 存储** (line 522-526): `kp_data['fsdc_loss']` 和 `kp_data['fsdc_stats']`。当 `kp_data is None` 时创建新 dict。正确。
5. **与 dual branch (STD-PR) 交互**: STD-PR (line 477-504) 在 FSDC 之前执行, 使用未 completed 的 `feat_map_detached`。这意味着 STD-PR 看到的是原始 (可能 occluded) 的 spatial tokens, 而 skeleton_head 看到的是 completed 的。设计上合理 — 两个分支职责不同。

### 测试分支 (line 647-660)

1. **条件检查** (line 647-648): `self.use_skeleton_gcn and pose_dict is not None and pose_test_feat != 'global'`。在使用 skeleton GCN 且非 global-only 模式时执行。正确。
2. **FSDC 调用** (line 651-656): 使用 `featmaps[-1]` (非 detached, 测试时无需)。条件 `use_fsdc and scene_heatmaps is not None`。正确。
3. **传入 skeleton_head** (line 658-660): `feat_for_gcn` 为 completed features。正确。
4. **注意**: 测试分支不影响 global feature (GAP pooling 在更上方已完成)。FSDC 只影响 part/keypoint features。这与设计意图一致。

---

## 完整审查: processor.py FSDC loss

**line 881-891**:
1. **条件检查**: `POSE_FSDC=True and kp_data is not None and 'fsdc_loss' in kp_data`。三重保护, 安全。
2. **Loss 计算**: `loss = loss + fsdc_weight * fsdc_loss`。weight 默认 0.5, 可配置。
3. **日志**: 记录 `fsdc_recon` 和 `fsdc_mask_ratio`。足够诊断。
4. **`_loss_details` 传递**: 正确获取和更新。

---

## 完整审查: config/defaults.py

**line 199-205**: 6 个 FSDC 配置键, 全部有合理默认值:
- `POSE_FSDC = False` — 默认关闭, 不影响已有实验
- `POSE_FSDC_LAYERS = 2` — 轻量
- `POSE_FSDC_HEADS = 8` — 768/8=96 head_dim, 合理
- `POSE_FSDC_MASK_RATIO = 0.3` — MAE 标准
- `POSE_FSDC_NOISE_STD = 0.1` — 适中
- `POSE_FSDC_WEIGHT = 0.5` — 与其他辅助 loss 一致

**安全性**: 所有键以 `POSE_FSDC` 前缀, 无命名冲突。`getattr(..., default)` 模式确保向后兼容。

---

## 优化器参数覆盖检查

`solver/make_optimizer.py` 遍历 `model.named_parameters()`, `FeatureDenoiser` 作为 `self.feature_denoiser` 注册为子模块, 其参数 (mask_token, pos_embed, mask_embed, decoder layers, output_proj, norm) 全部自动包含在优化器中。**无遗漏。**

---

## 残留 Issues 状态

| Issue | 状态 | 说明 |
|-------|------|------|
| C1 残差连接 | **已修复** | `masked_tokens + output_proj(decoded)` |
| C2 测试时 FSDC | **已修复** | 在 skeleton_head 前调用 |
| C3 测试残差 | **已修复** | 使用 `masked_tokens` |
| H1 Decoder self-attn 冗余 | Acknowledged | 非阻塞, 增加计算但无 bug |
| H2 Part loss → denoiser | Acknowledged | 实际是 desired (end-to-end) |
| H3 全可见 fallback | **已修复** | per-sample random mask fallback |
| M1 per-sample 循环 | Non-blocking | 小 tensor 操作, 影响微小 |
| M2 多余 noise 生成 | Non-blocking | 无功能影响 |
| M3 AMP 精度 | Non-blocking | threshold 计算在小 tensor 上, 风险低 |
| M4 测试 threshold | Non-blocking | 可在实验中调整 |

---

## 额外检查: 训练/测试对称性

- 训练: body-part mask → 替换为 mask_token + noise → denoise → recon loss + downstream loss
- 测试: pose-confidence mask → 替换为 mask_token (无 noise) → denoise → 用于 part matching

对称性合理: 训练时加 noise 增加鲁棒性, 测试时无 noise 是正确的。测试时 mask 逻辑 (低 heatmap activation → occluded) 与训练时 mask 逻辑 (body-part region mask) 在语义上对应 — 都是 "occluded 区域被 mask"。

---

## 额外检查: 日志充分性

训练日志输出:
- `fsdc_recon`: 重建 loss 值 — 可观察 denoiser 是否在学习
- `fsdc_mask_ratio`: 实际 mask 比例 — 可验证 mask 生成正常

建议 (非阻塞): 可增加 `fsdc_visible_tokens` 或 `fsdc_output_norm` 统计来观察 completed features 的分布变化。但现有日志足以判断模块是否工作。

---

## 结论

所有 3 个 Critical issues 和 1 个 High issue (H3) 已正确修复。代码逻辑、数据流、shape 变换、loss 计算、配置安全性均验证通过。剩余 issues (H1, H2, M1-M4) 均为 non-blocking, 已 acknowledged。

**审查通过** — 可以启动训练。
