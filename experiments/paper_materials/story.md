# 论文故事线（持续更新）

## 暂定标题

**OA-PAMS: Occlusion-Aware Part-Aggregated Multi-Scale Network for Occluded Person Re-Identification**

备选:
- Soft Pose Supervision with Visibility-Guided Feature Calibration for Occluded Person ReID
- Beyond Binary Visibility: Continuous Occlusion-Aware Part Learning for Person Re-Identification

## Motivation（为什么做这个）

### 现有问题
- 遮挡行人重识别是一个核心挑战：行人经常被物体或其他行人遮挡
- 遮挡导致两个问题：(1) 部分身体特征缺失 (2) 遮挡物引入噪声特征

### 现有方法的不足
1. **硬可见性**: 现有方法（KPR, PFD）将可见性二值化，"可见"部位全用、"不可见"部位全跳过。但实际上可见性是连续的——部分遮挡、自遮挡、模糊边界都很常见。
2. **粗糙的 Part 监督**: BPA (Body Part Attention) 使用姿态热图的 argmax 作为硬标签，丢失了热图的概率分布信息。在遮挡场景下，热图本身就不确定，硬标签会误导 part classifier。
3. **特征利用不充分**: 现有方法只是"跳过"遮挡部位，没有利用遮挡程度信息来自适应调整特征融合策略。

### 我们的洞察
- 姿态估计模型（如 ViTPose）的 visibility 向量和热图置信度包含了丰富的遮挡信息
- 这些连续的遮挡信号不应该被二值化，而应该在特征学习和匹配两个阶段都被充分利用
- SOLIDER 预训练的语义-外观解耦特征为结合遮挡信息提供了独特的基础

## 核心贡献（预计 3 点）

1. **Soft BPA (Body Part Attention)**: 首次在 part classifier 的监督中保留姿态热图的完整概率分布，通过温度调节的 soft label 替代 argmax 硬标签，使 part classifier 在遮挡边界区域学习更准确的部件归属。

2. **Visibility-Guided Feature Calibration (VGFC)**: 提出一个轻量模块，利用连续的 visibility 向量动态调整全局特征和各部件特征的融合权重。遮挡严重时依赖全局特征，遮挡轻微时利用精细的部件特征。

3. **Continuous Visibility-Weighted Part Distance**: 在推理阶段，用连续的 visibility score（而非二值 mask）加权各部件的距离贡献，使距离计算更好地适应不同程度的遮挡。

## 方法概述

```
Input Image → Swin-Tiny (SOLIDER pretrained)
  ├── Stage 0-3 features
  │
  ├── Multi-Scale Fusion (MSF) → Spatial Feature [B, D, 24, 8]
  │
  ├── Part Classifier → Part Probs [B, K+1, 24, 8]
  │   └── Supervised by Soft BPA Target (from pose heatmaps + temperature τ)
  │
  ├── Part Feature Extraction
  │   ├── Global: Stage 3 avg pool → [B, D]
  │   ├── Foreground: fg mask weighted pool → [B, D]
  │   └── K Parts: part mask weighted pool → [B, K, D]
  │
  ├── Visibility-Guided Feature Calibration (VGFC)
  │   ├── Input: part_vis [B, K], global_feat [B, D], part_feats [B, K, D]
  │   ├── MLP(vis_vector) → fusion weights α ∈ [0, 1]
  │   └── calibrated_feat = α * global + (1-α) * weighted_parts
  │
  └── Training Losses:
      ├── ID Loss (global + fg + per-part)
      ├── Part-Averaged Triplet (L2 norm + soft margin + vis-aware)
      ├── Soft BPA Cross-Entropy
      └── Push Diversity Loss

Inference Distance:
  d(q, g) = Σ_k  vis_q[k] * vis_g[k] * ||part_q[k] - part_g[k]|| / Σ_k vis_q[k]*vis_g[k]
```

## 实验证据链

{每个关键实验结果如何支撑我们的 story}
- 实验 exp001/002: Baseline 确认 → 证明提升来自我们的方法，非 baseline 变化
- 实验 exp003: Soft BPA vs Hard BPA → 证明保留概率信息有用
- 实验 exp004: NFC 后处理 → 证明 part-level 后处理有效
- 实验 exp005: OA-PAMS 全组件 → 证明系统性方法的整体效果
- 消融实验: 逐一移除 Soft BPA / VGFC / Continuous Vis Distance → 证明每个组件必要
- 可视化: Part attention map 对比 (soft vs hard BPA) → 直观展示效果

## 与 SOTA 对比的 narrative

目标: 在 Occ-Duke 上超过：
- SOLIDER Swin-Tiny baseline: 55.2% mAP (已确认)
- Pose-Swin best: 59.0% mAP (需要超过)
- 如果能达到 60%+ mAP，结合方法的轻量性（推理时不需要姿态模型），就是一个有竞争力的结果

## 待补充的实验 / 待解决的问题

- [ ] PAMS v9 完整训练结果
- [ ] Swin-Tiny baseline 重新确认
- [ ] Soft BPA 实现和验证
- [ ] VGFC 模块实现
- [ ] Continuous visibility distance 实现
- [ ] 完整消融实验
- [ ] t-SNE 可视化
- [ ] Part attention map 可视化
- [ ] 与 SOTA 方法的完整对比表
