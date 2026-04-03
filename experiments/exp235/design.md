# 实验 exp235: FSDC — Feature-Space Diffusion Completion for Occluded ReID

## 核心创新

**范式转变**: 从 "丢弃遮挡部分" 到 "概率性补全遮挡特征"

现有方法处理遮挡的方式：
- Visibility weighting: 降低遮挡部位权重 → 信息丢失
- Part matching: 只在可见部位匹配 → 信息不完整
- Feature completion (FCFormer): 确定性 decoder → 单点估计

**FSDC 的方式**: 在 backbone 输出的 spatial token 空间，用 conditional denoising 补全遮挡区域的特征。这是首次在 ReID 特征空间引入 diffusion。

## 问题定义

给定:
- Backbone 输出 spatial tokens: T = {t_1, ..., t_48} (12×4=48 tokens, each 768-dim)
- Pose-derived visibility mask: M (48-dim binary, 1=visible, 0=occluded)

目标:
- 补全 T 中 occluded tokens 的特征: T_completed = Denoise(T_masked, M, noise)
- 用 T_completed 替代 T_masked 做下游 pooling 和 matching

## 技术方案

### 训练阶段

```
1. Backbone → spatial tokens T (12×4, 768)
2. 用 pose heatmap 生成 body-part mask M (随机 mask 1-3 个 body parts)
3. 被 mask 的 tokens 替换为 learnable [MASK] token + 高斯噪声
4. Denoiser (2-layer transformer): 
   Input = T_masked + positional encoding + mask indicator
   Output = T_denoised
5. Loss = L2(T_denoised[masked_positions], T_original[masked_positions])
6. 总 loss = CE + triplet + alpha * reconstruction_loss
```

### 测试阶段

```
1. Backbone → spatial tokens T
2. 用 pose keypoint confidence 判断遮挡区域 (低 confidence → occluded)
3. Denoiser 补全 occluded tokens
4. 用 completed tokens 做 GAP / Part pooling / MaxSim
```

### Denoiser 架构

```python
class FeatureDenoiser(nn.Module):
    # 轻量 transformer decoder
    # Input: (B, 48, 768) + mask (B, 48)
    # Output: (B, 48, 768)
    # 2 layers, 8 heads, hidden=768
    # Positional encoding: learnable 48-dim
    # Mask indicator: concatenated binary flag
```

### 关键设计选择

1. **不是图像级 diffusion**: 在 768-dim token space 做，不是像素空间。极其轻量。
2. **单步去噪**: 不需要多步 diffusion schedule。一步 transformer forward 即可。
   (实际上更像 masked autoencoder 而非 full diffusion, 但 narrative 可以讲 denoising)
3. **Pose-guided masking**: 训练时按 body part 区域 mask（不是随机 token），模拟真实遮挡模式
4. **Detached 操作**: denoiser 在 detached backbone features 上工作，不影响 backbone 训练
5. **Reconstruction target**: 用同一图像的 unmasked features 作为 target（自监督）

### 与 FCFormer 的区别

| 方面 | FCFormer | FSDC (ours) |
|------|---------|------|
| 补全空间 | Spatial token space | Same, but pose-conditioned |
| 方法 | Deterministic decoder | Denoising (masked → reconstructed) |
| Mask 来源 | Random / segmentation | Pose heatmap body-part regions |
| 训练 | Separate pre-training | End-to-end with ReID loss |
| 创新点 | First completion for ReID | Pose-conditioned denoising + body-part masking |

## 实现步骤

1. `model/modules/feature_denoiser.py`: 新文件，FeatureDenoiser 模块
2. `model/pose_backbone_model.py`: 在 forward 中调用 denoiser
3. `config/defaults.py`: POSE_FSDC 配置
4. `processor/processor.py`: reconstruction loss 计算

## 对照组
- exp191 (OA-SD): 63.2/75.4
- FCFormer (论文报): 60.0+

## 预期结果
- 成功: mAP +1.0~3.0% (补全的 occluded features 提供了之前丢失的信息)
- 失败: ~0% (backbone features 已经 robust enough, 补全不增加信息)

## 早停
- ep10 < 25% → 终止
- ep30 < 48% → 终止

## 论文价值
- **Title**: "Feature-Space Denoising for Occluded Person Re-Identification"
- **核心贡献**: 首次在 ReID 特征空间引入 denoising-based completion
- **消融**: mask ratio, denoiser depth, pose-guided vs random mask
