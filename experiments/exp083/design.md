# 实验 exp083: Pose-Guided Feature Inpainting (PGFI)

## 动机
- 遮挡 ReID 的核心问题: 被遮挡的身体部位没有特征 → 特征不完整 → 检索失败
- FCFormer 用 feature completion decoder 恢复遮挡特征（TPAMI 2024, Occluded-Duke SOTA 之一）
- 我们的 PSG/PAA 只做 "suppress background" 和 "inject pose info"，没有做 "recover occluded"
- **新方向**: 用 pose heatmap 识别遮挡区域，用可见区域的特征"推断"遮挡区域特征

## 创新点 / 核心想法
- **Pose-Guided Feature Inpainting**: 在 Stage 3 features 上做 pose-aware inpainting
- 可见区域（heatmap 响应高）: 保留原始特征
- 遮挡区域（heatmap 响应低）: 用 learned inpainter 从可见特征推断
- inpainter 是一个轻量 Conv 网络，输入是 masked feature map + pose heatmap

## 技术方案

### 架构
```
Stage 3 features (B, 48, 768) [PSG+PAA enhanced]
        ↓
1. 从 pose heatmap 生成 visibility mask: vis = sigmoid(heatmap).max(dim=1)  (B, 1, H, W)
2. 分离: feat_visible = feat * vis, feat_masked = feat * (1 - vis)
3. Inpainter:
   input = cat(feat_visible, heatmap_resized)  # (B, 768+17, H, W)
   inpainted = Conv(768+17, 256) → ReLU → Conv(256, 768) → zero_init
4. 合并: feat_final = feat_visible + (1-vis) * inpainted
   → 可见区域保持不变，遮挡区域用 inpainted 特征填充
```

### 关键设计
1. **只修改遮挡区域**: 通过 (1-vis) mask 确保可见区域特征不变
2. **Pose-conditioned inpainting**: inpainter 接收 heatmap 作为条件，知道"应该推断哪个身体部位"
3. **Zero-init output**: 初始时 inpainted = 0，等效于不修改
4. **在 PSG+PAA 之后、GAP 之前应用**: 利用已经 pose-enhanced 的特征

### 参数
- Conv(785, 256): ~200K
- Conv(256, 768): ~200K
- 总计 ~400K (与 GCN 相当)

### 位置
- 在 Stage 3 最后一个 block 的输出上应用（不是在每个 block 后）
- 在 GAP 之前

## 与已有方法的区别
- vs PSG: PSG suppress 背景，PGFI recover 遮挡
- vs PAA: PAA inject pose info，PGFI 用 pose 指导 inpainting
- vs SGMKC (exp048): SGMKC 在 GCN keypoint 空间做，PGFI 在 feature map 空间做
- vs FCFormer: FCFormer 用 transformer decoder，PGFI 用简单 conv inpainter

## 对照
- exp066 PAA = 61.6%/74.2%
- exp030a 3-seed = 60.73%/72.57%

## 预期
- 如果有效: 遮挡特征被恢复 → mAP/R1 提升（尤其在重度遮挡样本上）
- 如果无效: 12×4 分辨率上 inpainting 太粗糙 / conv inpainter 容量不够
