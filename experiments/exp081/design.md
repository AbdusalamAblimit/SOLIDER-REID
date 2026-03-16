# 实验 exp081: Pose-Query Transformer Decoder (PQTD)

## 动机
- GCN branch 对 fusion 有稳定 +1.40% 增益，但 GCN 本身的表达能力有限（2 层图卷积，固定拓扑）
- exp053 XCAD 尝试过单层 cross-attention 替代 GCN，效果弱（-1.03%）
- DPEFormer/ProFD/FCFormer 等 SOTA 都使用 multi-layer transformer decoder 做 part feature extraction
- **假设**: 多层 transformer decoder + pose-guided learnable queries 可以替代 GCN 并获得更强的 branch 特征

## 创新点 / 核心想法
- 用 **pose heatmap 初始化的 learnable queries** 做 multi-layer cross-attention decoder
- 每个 query 代表一个身体部位（5 parts: head, shoulders, arms, hips, legs）
- Decoder 从 backbone feature map 中提取 part-specific 信息
- 这是 "pose-guided query" 的范式——与 KPR 的 keypoint prompt 类似但机制不同

## 技术方案

### 架构
```
Backbone features (B, 12×4, 768) [PSG+PAA enhanced]
        ↓
  ┌─────────────────────────────────────┐
  │ Pose-Query Transformer Decoder       │
  │                                      │
  │ 5 learnable part queries (5, 256)    │
  │ + pose position encoding             │
  │ ↓                                    │
  │ Decoder Layer ×3:                    │
  │   Self-Attn(queries, queries)        │
  │   Cross-Attn(queries, backbone_feat) │
  │   FFN                                │
  │ ↓                                    │
  │ 5 part features (5, 256)             │
  │ ↓                                    │
  │ Concat → Linear(1280, 768) → feat   │
  └─────────────────────────────────────┘
        ↓
  Part ID + Part Triplet losses (detached from backbone)
  Test: equal_concat(global, decoder_feat)
```

### 关键设计
1. **Part queries** 是 learnable embeddings (5, 256)
2. **Pose position encoding**: 用 heatmap GAP 计算的 5-part encoding 加到 queries
3. **3 层 decoder**: 足够深来学到有意义的 part-specific 注意力模式
4. **Decoder dim = 256**（不是 768）：减少参数量和计算
5. **Backbone features 通过 linear projection 降到 256 维**
6. **Output 通过 concat(5 parts) → linear → 768 dim part feature**

### 参数估算
- Decoder layer: ~400K × 3 = ~1.2M
- Query embeddings: 5 × 256 = 1.3K
- Projections: ~600K
- Classifier: ~600K
- 总计: ~2.5M params（比 GCN 的 400K 多，但比 PDS 的 6.3M 少）

### 与 exp053 XCAD 的区别
- exp053: 单层 cross-attention, 17 keypoint queries, 无 self-attention
- exp081: **3 层 decoder, 5 part queries (不是 17 kp), self+cross attention, learnable queries + pose PE**

## 对照
- exp066 PAA (PSG+GCN+PAA) = 61.6%/74.2%
- exp030a (PSG+GCN) 3-seed = 60.73%/72.57%
- 消融变量: 用 PQTD 替换 GCN branch

## 预期
- 如果成功: branch feature 更强 → equal_concat > 61.6%
- 如果失败: decoder 需要更多数据/epoch 收敛，15K 训练图不够
