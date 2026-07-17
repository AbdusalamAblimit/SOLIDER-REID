# 实验 exp054: Pose-Guided Attention Masking (PGAM)

## 动机
- PSG 通过乘性门控调制特征值，但遮挡物 token 仍然参与自注意力计算，会通过注意力机制"污染"人体 token 的表征
- FPC (AAAI 2024) 证明 token 稀疏化在遮挡 ReID 中有 +8.6% mAP 的巨大潜力
- 但 FPC 用学习的 CLS token 注意力做 token 选择（无 pose），我们有更直接的信号：pose heatmap
- PGAM 在自注意力内部阻断非人体 token 与人体 token 之间的注意力路径

## 创新点 / 核心想法
- **核心假设**: 在 Swin Stage 3 的自注意力中，用 pose heatmap 硬掩码阻断非人体 token 参与注意力，可以减少遮挡物对人体特征的污染
- **与 PSG 的区别**: PSG 调制特征幅值（所有 token 保留，值变小）；PGAM 阻断注意力路径（token 保留但不参与注意力交互）
- **与 PAB/KP-RPE 的区别**: PAB/KP-RPE 是软性加法偏置（微调注意力权重）；PGAM 是硬性掩码（完全阻断）。效果量级完全不同
- **零额外参数**: PoseAttnMask 模块无可学习参数，纯计算

## 技术方案
- 新增 `model/modules/pose_attn_mask.py`: PoseAttnMask 模块
  - 输入: scene_heatmaps (B, 17, H_hm, W_hm), hw_shape
  - 计算: max(heatmaps, dim=1) → resize → threshold → 非body位置设-50
  - 输出: pose_bias_map (B, num_heads, H_feat, W_feat)
  - 通过 ShiftWindowMSA 的 `pose_bias_map` 路径传递
  - 在 WindowMSA 中经过 additive decomposition: bias(i,j) = val[i] + val[j]
  - 非body pair: -50 + -50 = -100 → 完全阻断
  - body↔非body: 0 + -50 = -50 → 基本阻断
  - body↔body: 0 + 0 = 0 → 正常注意力
- 修改 `model/pose_backbone_model.py`: 在 PSG-only 路径中，先生成 PGAM mask 传给 block，再应用 PSG gate
- 新增 config: `POSE_ATTN_MASK`, `POSE_ATTN_MASK_THRESHOLD`

## 数据流
```
Input → Swin Stage 0-2 → Stage 3:
  For each block:
    1. PGAM: heatmap → body mask → attention bias → block(x, pose_bias_map=bias)
    2. PSG: x = x * (1 + gate(heatmap))
  → GAP → global feat
  → GCN → keypoint feats
  → concat → equal_concat output
```

## 预期结果
- **如果假设成立**: PGAM + PSG 互补 → mAP 提升 0.5-1.5%（PSG 调幅值，PGAM 阻污染）
- **如果失败**: 硬掩码可能过于激进（丢失了遮挡物的语义上下文信息）或与 PSG 冗余
- **最可能原因**: 12×4 的特征图上 7×7 window attention 仅覆盖 2 个 window，每个 window 内非body token 数量可能很少，掩码效果有限

## 对照组
- Baseline 对照: exp030a (PSG+GCN, 无 PGAM)
- 消融变量: 仅增加 PGAM attention masking
