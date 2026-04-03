# exp235 FSDC 审查报告

审查人: Claude Opus 4.6
审查日期: 2026-04-03
审查范围: design.md, feature_denoiser.py, pose_backbone_model.py FSDC集成, processor.py FSDC loss, config/defaults.py

---

## Critical Issues

### C1: 残差连接导致重建目标退化为零（feature_denoiser.py:169）

**严重程度: Critical — 训练完全无效**

训练路径中:
- `target = spatial_tokens.detach().clone()` (line 148) — 保存原始特征
- `masked_tokens` 是替换了 masked 位置的副本 (line 151-158)
- 但 `output = spatial_tokens + self.output_proj(decoded)` (line 169) 使用的是**未修改的** `spatial_tokens`

因此对于 masked positions:
```
loss = MSE(spatial_tokens[masked] + output_proj(decoded)[masked], target[masked])
     = MSE(original_feature + output_proj(decoded), original_feature)
     = MSE(output_proj(decoded), 0)
```

由于 `output_proj` 是 zero-init, 初始 loss 接近 0, 并且 denoiser 学到的最优解是始终输出零 — 即不做任何补全。整个重建训练失效。

**修复方案**: 将 line 169 改为:
```python
output = masked_tokens + self.output_proj(decoded)
```
这样 masked 位置的重建目标才是: `mask_token + noise + correction → original`, 即 denoiser 真正学习从 corrupted 到 original 的映射。

### C2: 测试时 FSDC 未被调用（pose_backbone_model.py:647-652）

**严重程度: Critical — 测试时补全功能不存在**

FSDC forward 只在 `self.training` 分支中被调用 (line 508)。在测试分支 (line 599+), skeleton_head 直接接收 `featmaps[-1]`，从未经过 denoiser。

设计文档明确写: "测试阶段: Denoiser 补全 occluded tokens → 用 completed tokens 做 GAP / Part pooling / MaxSim"。但这完全没有实现。

**修复方案**: 在测试分支 (line 647-652 之前) 加入 FSDC 调用:
```python
if getattr(self, 'use_fsdc', False) and scene_heatmaps is not None:
    feat_for_part = featmaps[-1]  # no detach at test time
    B_d, C_d, H_d, W_d = feat_for_part.shape
    spatial_tokens = feat_for_part.flatten(2).transpose(1, 2)
    completed, _, _ = self.feature_denoiser(spatial_tokens, scene_heatmaps, fH=H_d, fW=W_d)
    feat_for_part = completed.transpose(1, 2).reshape(B_d, C_d, H_d, W_d)
else:
    feat_for_part = featmaps[-1]
# 然后用 feat_for_part 而非 featmaps[-1] 传入 skeleton_head
```

### C3: 测试时残差连接同样有问题（feature_denoiser.py:217）

**严重程度: Critical（前提是 C2 被修复后才会触发）**

测试路径 line 217: `output = spatial_tokens + self.output_proj(decoded)` 同样在 occluded 位置使用原始（可能是噪声/无意义的）特征做残差基础。应与训练路径保持一致，使用 `masked_tokens` 作为残差基础:
```python
output = masked_tokens + self.output_proj(decoded)
```

---

## High Issues

### H1: TransformerDecoder 的 self-attention 冗余（feature_denoiser.py:165）

`self.decoder(input_tokens, input_tokens)` 中 `tgt == memory`。`TransformerDecoderLayer` 包含 self-attention + cross-attention 两个子层, 当 tgt==memory 时 cross-attention 退化为另一次 self-attention。
- 计算量翻倍但无明确收益
- 建议: 使用 `nn.TransformerEncoder` 替代 (单 self-attention), 或将 visible tokens 作为 memory, masked tokens 作为 tgt 实现真正的 cross-attention completion

### H2: 部分分支 loss 会反传到 denoiser（pose_backbone_model.py:514-519）

`feat_map_detached` 被替换为 denoiser 输出 (有梯度), 然后传入 skeleton_head。skeleton_head 产生的 CE/triplet loss 会反传到 denoiser 参数。

这与设计文档 "Detached 操作: denoiser 在 detached backbone features 上工作，不影响 backbone 训练" 的意图不完全一致。虽然 backbone 确实不受影响, 但 denoiser 同时接收:
1. 重建 loss (显式)
2. 部分分支 ID/triplet loss (隐式, 通过 completed features)

这可能是有益的 (end-to-end training), 但应在设计文档中明确。如果只想用重建 loss 训练 denoiser, 需要在 line 514 之后对 `feat_map_detached` 再做一次 `.detach()`。

### H3: _generate_body_part_mask 可能产生全可见 mask

当所有选中 body part 的 heatmap activation 都低于 threshold 时, mask 保持全 True (全可见)。这会导致 `recon_loss = 0.0` (line 179), 浪费这个 batch sample 的训练信号。

建议加一个 fallback: 如果 mask 后 masked token 数量 < 1, 退回到 random mask。

---

## Medium Issues

### M1: per-sample Python 循环效率低（feature_denoiser.py:101-112）

`_generate_body_part_mask` 包含 `for b in range(B)` 循环, 每个 sample 内还有 `for part_idx` 循环。B=64 时最多 64*3=192 次小循环。虽然每次操作在 (12,4) tensor 上很快, 但可以考虑向量化。

非阻塞问题, 但如果训练速度明显受影响应优化。

### M2: noise 添加到所有 tokens（feature_denoiser.py:153）

`noise = torch.randn_like(masked_tokens) * self.noise_std` 生成了所有 token 的 noise, 但只有 masked 位置使用。visible 位置的 noise 被 `torch.where` 丢弃。浪费少量计算, 无功能问题。

### M3: 缺少 AMP autocast 标注

`_generate_body_part_mask` 中的 `F.interpolate` 在 AMP 下可能以 float16 执行。`part_hm.mean() + 0.5 * part_hm.std()` 在 float16 下精度可能不足导致 threshold 不稳定。建议在该函数内加 `@torch.cuda.amp.custom_fwd(cast_output=torch.float32)` 或手动 `.float()` 转换。

### M4: 测试时 threshold 过于激进（feature_denoiser.py:199）

`mask = hm_flat > threshold * 0.5` 意味着 heatmap activation 低于均值一半的 token 就被认为 occluded。对非遮挡图像, 背景区域也会被大量 mask, 导致不必要的 denoising。考虑使用绝对 threshold 或更保守的相对 threshold。

---

## Low Issues

### L1: 设计文档标题提及 "Diffusion" 但实现是 Masked Autoencoder

设计文档承认 "实际上更像 masked autoencoder 而非 full diffusion"。名称 "FSDC" 中的 "Diffusion" 可能在论文审稿时被质疑。建议在内部文档中更精确地称为 "Feature-Space Masked Completion" (FSMC)。

### L2: `torch.tensor(0.0, device=device)` 无 requires_grad（feature_denoiser.py:179）

当没有 masked positions 时返回一个无梯度的 0 tensor。在 processor 中 `loss + fsdc_weight * fsdc_loss` 不会报错但 `.item()` 在 details 中会是 0.0。这是正确行为但可加注释说明。

---

## 配置安全检查

- `POSE_FSDC = False` (默认关闭): 安全, 不影响已有实验
- `POSE_FSDC_WEIGHT = 0.5`: 合理默认值
- 所有新配置键都使用 `getattr` 带默认值访问: 安全
- 无配置文件被创建 — 实验还需创建 config yaml

---

## 总结

发现 3 个 Critical 级别问题, 全部必须修复后才能训练:

1. **残差连接 bug** — denoiser 学习输出零而非重建特征, 整个模块无效
2. **测试时 FSDC 未调用** — 核心功能缺失
3. **测试时残差连接同样有 bug** — 与训练保持一致需要修复

此外有 3 个 High 级别问题需要修复或明确文档化。

**审查结论: 未通过 — 需修复 C1-C3 后重新审查**
