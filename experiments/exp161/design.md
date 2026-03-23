# 实验 exp161: STD-PR (Structural Token Decomposition with Pose-guided Routing)

## 范式级创新

### 核心范式转变
```
传统: Image → Backbone(spatial tokens 12×4) → GAP/Part Pooling → Feature
STD-PR: Image → Backbone Stage 0-2 (spatial tokens) → [Structural Routing] → Stage 3 (structural tokens K个) → Feature
```

**在 Swin Stage 3 之前，用 pose-guided cross-attention 把 spatial tokens 不可逆转换为 K 个 body-part structural tokens。** Stage 3 不再对 spatial grid 做 window attention，而是对 K 个 structural tokens 做 self-attention。

### 与现有工作的关键区分

| | PAFormer | KPR | PAT | **STD-PR (ours)** |
|---|---------|-----|-----|------------------|
| Backbone | ViT | Swin | ViT | **Swin** |
| Part tokens 位置 | 全部 12 层 | 输入层 prompt | 全部层 | **Stage 2→3 之间** |
| Spatial tokens 是否保留 | 是 | 是 | 是 | **否（不可逆转换）** |
| Pose guidance | Heatmap supervision | Input prompt | 无 | **Heatmap-guided cross-attn** |

### 为什么让人眼前一亮

1. **明确的 representational commitment**: 在某个层之后，模型不再"看像素"而是"看身体部件"
2. **解决了 12×4 分辨率的根本瓶颈**: spatial grid 上做 part pooling 精度受限，structural tokens 直接是 part-level
3. **与 Swin 的 hierarchical 设计天然契合**: Swin 的 stage 之间有 patch merging 做空间下采样，我们加一步 "spatial→structural" 转换

## 技术方案

### 1. Structural Routing Layer（插在 Stage 2 → Stage 3 之间）

```python
class StructuralRoutingLayer(nn.Module):
    """Convert spatial tokens to structural body-part tokens via pose-guided cross-attention."""

    def __init__(self, spatial_dim, num_parts=6, num_heads=8):
        # K learnable part query embeddings
        self.part_queries = nn.Parameter(torch.randn(num_parts, spatial_dim))
        # Cross-attention: part queries attend to spatial keys/values
        self.cross_attn = nn.MultiheadAttention(spatial_dim, num_heads, batch_first=True)
        # Pose bias: heatmap → attention bias for each part query
        self.pose_bias_proj = nn.Conv2d(17, num_parts, kernel_size=1)
        # FFN after cross-attention
        self.ffn = nn.Sequential(
            nn.Linear(spatial_dim, spatial_dim * 4),
            nn.GELU(),
            nn.Linear(spatial_dim * 4, spatial_dim),
        )
        self.norm1 = nn.LayerNorm(spatial_dim)
        self.norm2 = nn.LayerNorm(spatial_dim)

    def forward(self, spatial_tokens, hw_shape, scene_heatmaps):
        """
        Args:
            spatial_tokens: (B, H*W, C) from Stage 2 output
            hw_shape: (H, W) spatial dimensions
            scene_heatmaps: (B, 17, hm_H, hm_W) pose heatmaps
        Returns:
            structural_tokens: (B, K, C) body-part tokens
        """
        B = spatial_tokens.shape[0]
        H, W = hw_shape

        # Expand part queries for batch
        queries = self.part_queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, C)

        # Compute pose-guided attention bias
        hm_resized = F.interpolate(scene_heatmaps, size=(H, W), mode='bilinear')
        pose_bias = self.pose_bias_proj(hm_resized)  # (B, K, H, W)
        pose_bias = pose_bias.view(B, K, H*W)  # (B, K, H*W) as attention bias

        # Cross-attention with pose bias
        # queries attend to spatial tokens, biased by pose heatmaps
        structural_tokens = self.cross_attn(
            query=self.norm1(queries),
            key=spatial_tokens,
            value=spatial_tokens,
            attn_mask=None,  # pose_bias 作为 additive bias
        )[0] + queries  # residual

        # FFN
        structural_tokens = structural_tokens + self.ffn(self.norm2(structural_tokens))

        return structural_tokens  # (B, K, C)
```

### 2. Modified Stage 3: Self-Attention on Structural Tokens

Stage 3 的两个 SwinBlock 改为在 K 个 structural tokens 上做 **standard self-attention**（不是 window attention，因为 K 很小）。

```python
# Original Stage 3: window attention on 12×4=48 spatial tokens
# STD-PR Stage 3: self-attention on K=6 structural tokens
for block in stage3.blocks:
    structural_tokens = block.self_attention(structural_tokens)  # (B, K, C)
    # PSG gate 也可以在这里用（用 heatmap 对 structural tokens 做 gate）
```

### 3. 分组方案

K=6 body parts (不是 17 个 keypoint，因为太多 token 会 dilute attention):
- head: kp 0-4 (nose, eyes, ears)
- torso: kp 5-6, 11-12 (shoulders, hips)
- left_arm: kp 5, 7, 9
- right_arm: kp 6, 8, 10
- left_leg: kp 11, 13, 15
- right_leg: kp 12, 14, 16

### 4. 训练

- Structural tokens 经过 GAP → global feature（与原 global 等价）
- 每个 structural token 也可以独立做 part-level ID classification
- 与 GCN 的关系：STD-PR **替代** GCN branch（structural tokens 已经是 body-part level）

### 5. Test-time

- 6 个 structural tokens 做 equal_concat（类似当前 global + GCN）
- 也可以做 MaxSim（6×6 matching）

## 预期效果

- 如果成功：解决了"12×4 空间分辨率上做 part pooling 精度不足"的根本问题
- structural tokens 的每个 token 都"知道"自己是哪个 body part
- 遮挡场景下：被遮挡的 part token 自然弱化（cross-attention 的 pose bias 让它收集到的信息少）

## 风险

1. Stage 3 从 48 tokens window-attn 变成 6 tokens self-attn，容量大幅减少
2. Cross-attention 的 pose bias 可能不够精确
3. 需要足够 epoch 来训练 cross-attention（exp081 decoder 120ep 不够）
4. 预训练 Stage 3 的权重无法直接复用（从 window-attn 变成 self-attn）

## 对风险 4 的解决方案

**不完全替换 Stage 3**。保留原 Stage 3 对 spatial tokens 的处理（不动），在 Stage 3 之后额外加 Structural Routing Layer。这样：
- 原 backbone 完全不变（保留预训练权重）
- Structural Routing 是 backbone 之后的第一个新模块
- structural tokens 替代 GCN 的 keypoint pooling

这更安全，也更像 "GCN 的替代方案" 而非 "backbone 的修改"。

## 最终架构

```
Image → Swin Stage 0-2-3 (unchanged, spatial tokens) → [Structural Routing Layer] → K structural tokens
                                                                                    ↓
                                                                              Part Self-Attn (2 layers)
                                                                                    ↓
                                                                              K part features
                                                                                    ↓
Global feature (GAP on spatial) + Part features (structural tokens) → equal_concat
```

这本质上是**用 pose-guided cross-attention decoder 替代 GCN 的 keypoint point-sampling + graph propagation**。

区别于 exp063 PTD（纯 decoder，120ep 不够收敛）：
- STD-PR 的 cross-attention 有 pose heatmap bias（聚焦到正确位置）
- Part queries 有 learnable initialization（不是 random）
- 只有 1-2 层 cross-attention + 1-2 层 self-attention（比 3-layer decoder 轻）
