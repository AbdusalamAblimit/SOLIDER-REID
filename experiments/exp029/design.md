# 实验 exp029: PSG + Pose-Weighted Pooling (PWP)

## 动机
- exp007 PSG 是目前单一方法最佳 (mAP 58.3%)，在 backbone 特征形成阶段注入 pose 信息
- 但最终的 Global Average Pooling (GAP) 仍然等权重平均所有 token（包括背景和遮挡区域）
- 在遮挡 ReID 中，大量 token 对应遮挡物或背景，GAP 会稀释有效的人体特征
- 假设：如果用 pose 热图指导 pooling，仅聚焦于人体可见部位的 token，可以产生更纯净的 global feature
- 与 Part Pooling (exp001/008) 的关键区别：**不引入任何额外分类器或 loss**，不改变梯度流，仅改变 pooling 权重

## 创新点 / 核心想法
**核心假设：PSG 让 backbone 产生了 pose-aware 的 token 特征，但 GAP 丢弃了空间选择信息。用 pose 热图的空间响应作为 pooling 权重，可以在不增加任何参数的情况下过滤噪声 token，产生更纯净的 global 特征。**

这是 PSG ("让 backbone 知道人在哪") + PWP ("让 pooling 也知道人在哪") 的完整闭环：
- PSG: 特征形成阶段的 pose 引导 → 更好的 token 特征
- PWP: 特征聚合阶段的 pose 引导 → 更纯净的 global 特征

## 技术方案

### 修改文件
1. **`config/defaults.py`**: 新增 `POSE_WEIGHTED_POOL = False`（默认关闭，不影响其他实验）
2. **`model/pose_backbone_model.py`**: 在 forward() 中替换 GAP 为 pose-weighted pooling

### 数据流
```
输入: featmap (B, 768, 24, 8) + scene_heatmaps (B, 17, H, W)

1. scene_heatmaps → resize 到 (24, 8)
2. → sigmoid → (B, 17, 24, 8)  [0, 1] 范围
3. → max over channels → body_mask (B, 1, 24, 8)  [0, 1]
4. → 加 epsilon 防止全零
5. weighted_pool: (featmap * body_mask).sum(dim=(2,3)) / body_mask.sum(dim=(2,3))
6. → global_feat (B, 768)

替代标准 GAP:
   GAP: featmap.mean(dim=(2,3)) = featmap.sum / (24*8)
   PWP: (featmap * body_mask).sum / body_mask.sum
```

### 关键设计选择
- **sigmoid on raw heatmap**: PSG 也用 sigmoid，保持一致性。raw heatmap 范围 [-5, +20]，sigmoid 将高响应区→~1，低响应区→~0
- **max over 17 channels**: 产生"任意身体部位存在"的 map。不区分部位，只关心"是否为人体"
- **无额外参数**: 完全利用已有的 pose 热图信息，不增加任何可学习参数
- **soft weighting（非 hard mask）**: 不使用阈值截断，保留 background token 的微弱贡献（regularization 效果）

### 关键超参数
- 无新增超参数
- 其余所有参数与 exp007 (PSG) 完全相同

## 预期结果
- **如果假设成立**: mAP > 58.3%（超过 PSG-only），因为遮挡图中的噪声 token 被降权
- **如果中性 ≈ exp007**: PSG 已经让 backbone 内部适应了遮挡，后续 pooling 权重不重要
- **如果 < exp007**: pose-weighted pooling 破坏了特征分布，BN 层无法适应
- **最可能失败原因**: PSG 已经足够好地处理了遮挡信息，在 backbone 内部完成了 token 重要性加权，后续 pooling 加权是冗余的

## 对照组
- **直接对照**: exp007 PSG-only (mAP 58.3%, R1 67.9%)
- **消融变量**: 仅 pooling 方式从 GAP → PWP，其他完全不变
