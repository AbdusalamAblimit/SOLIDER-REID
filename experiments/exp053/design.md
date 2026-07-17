# 实验 exp053: Pose-Guided Cross-Attention Decoder (PGCAD)

## 动机
- 7 次在 PSG+GCN 框架上的增量修改（5 辅助 loss + 2 注意力偏置）全部中性/失败
- 发现的核心规律：**直接改变特征加工方式（如 PSG 乘性门控）有效，添加额外训练信号（如辅助 loss）无效**
- 当前 GCN branch 的局限：keypoint pooling 从 backbone 特征中提取 per-keypoint features，然后通过 2 层 GCN 传播。但：
  1. Keypoint pooling 只用 single token（bilinear sampling），信息量有限
  2. GCN 只在 17 个 keypoint 之间传播，无法回到 backbone 特征空间获取更多上下文
  3. 整个 branch 是单向的：backbone → keypoint → GCN → feature，没有反馈

## 创新点 / 核心想法
**用 Cross-Attention Decoder 替换 GCN branch**：将 17 个关键点位置编码为 learnable query tokens，对 Stage 3 feature map 做 cross-attention，直接从 backbone 特征中解码结构化 part 特征。

核心假设：Cross-Attention 比 keypoint bilinear sampling + GCN 能提取更丰富的 per-keypoint 特征，因为：
1. 每个 keypoint query 可以 attend to 整个 feature map（而非单个空间位置）
2. 自然处理遮挡：遮挡区域的 keypoint query 会学到从可见区域收集补偿信息
3. 解码器与 backbone 形成 encode-decode 结构，更符合现代视觉架构范式

## 技术方案

### 模块设计: PoseCrossAttentionDecoder
```
输入:
  - backbone_features: (B, H*W, C) = (B, 48, 768) — Stage 3 输出
  - keypoints: (B, 17, 2) — COCO keypoint 坐标
  - scores: (B, 17) — keypoint 置信度

流程:
  1. Keypoint Position Encoding:
     - 将 17 个 keypoint 坐标映射到 feature map 空间
     - 通过 MLP 生成 keypoint position embeddings: (B, 17, C)

  2. Learnable Keypoint Queries:
     - 17 个 learnable query tokens: (17, C)
     - 加上 position encoding: queries = learned_queries + pos_encoding

  3. Cross-Attention Decoder (1-2 layers):
     - Layer structure: CrossAttention → LayerNorm → FFN → LayerNorm
     - Query: keypoint queries (B, 17, C)
     - Key/Value: backbone features (B, H*W, C)
     - Output: decoded keypoint features (B, 17, C)

  4. Confidence-Weighted Pooling:
     - 用 keypoint scores 加权 pool 17 个 features 为 part feature
     - 或者按 body region 分组(头/躯干/左臂/右臂/左腿/右腿)

输出:
  - kp_features: (B, 17, C) 或 pooled (B, C)
```

### 与现有架构的集成
- 替换 `SkeletonGCN` 模块为 `PoseCrossAttentionDecoder`
- PSG 仍然保留（backbone 级 pose 注入）
- 训练 loss 不变：per-keypoint ID loss + per-keypoint triplet loss（与 GCN 相同）
- 测试时特征提取方式不变：global concat part

### 关键超参数
- Decoder layers: 1（最简版本，后续可增加）
- Attention heads: 8
- FFN hidden dim: 768*4 = 3072（标准 transformer FFN ratio）
- Dropout: 0.1
- Position encoding: MLP(2→256→768) + ReLU

### 修改文件
1. 新增: `model/modules/pose_cross_attn_decoder.py` — 核心模块
2. 修改: `model/pose_backbone_model.py` — 集成解码器
3. 修改: `config/defaults.py` — 添加配置项
4. 新增: `configs/occluded_duke/pose_psg_xcad.yml` — 实验配置

### 参数量估算
- 1 层 cross-attention decoder:
  - Q/K/V projections: 3 × 768 × 768 = ~1.77M
  - Output projection: 768 × 768 = ~0.59M
  - FFN: 768 × 3072 + 3072 × 768 = ~4.72M
  - LayerNorm: ~3K
  - Total per layer: ~7.1M
- Learnable queries: 17 × 768 = ~13K
- Position MLP: 2 × 256 + 256 × 768 = ~197K
- **Total: ~7.3M** (vs GCN ~0.5M)

显存估算：7.3M params × 4 bytes = ~29MB 参数 + attention maps (B×17×48) + gradients
在 8GB batch 使用量上增加约 500MB-1GB，应在 24GB 3090 范围内。

## 预期结果
- **如果假设成立**: mAP 提升 +1-2%（因为 cross-attention 能提取比 GCN 更丰富的 per-keypoint 特征）
- **如果失败**: 最可能原因是：
  1. 参数量过大导致过拟合（702 个 ID 的小数据集）
  2. Cross-attention 无法聚焦到正确的 body part region
  3. Position encoding 不够精确导致 query 与错误区域对齐

## 对照组
- Baseline 对照: exp030a（PSG + GCN, equal_concat）60.73% mAP / 72.57% R1 (3-seed mean)
- 消融变量: 用 Cross-Attention Decoder 替换 Skeleton GCN，其余保持不变

## 风险与缓解
1. **过拟合风险**: 7.3M 新参数 vs GCN 的 0.5M → 缓解：dropout 0.1, weight decay, 可考虑只用 1 层减少参数
2. **显存风险**: 估算在 24GB 范围内 → 缓解：如果 OOM，减少 FFN hidden dim 或 attention heads
3. **训练稳定性**: Cross-attention 初始化随机可能导致早期训练不稳定 → 缓解：使用 Xavier 初始化，较低初始 LR
