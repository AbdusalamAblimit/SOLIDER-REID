# Paper 8: BPBreID - Body Part-Based ReID
**来源**: WACV 2023
**仓库**: https://github.com/VlSomers/bpbreid
**核心**: 可学习注意力的 Part-Based ReID + 可见性感知

## 可拆解模块清单

### M1: Learnable Attention (PixelToPartClassifier)
- Conv1x1: feat_dim → K+1 → softmax → 部件概率分布
- 无需人工标注, 完全端到端学习
- **移植可行性**: 高 | **显存**: <0.1G

### M2: Visibility Scores
- 连续模式: vis = max(prob_map) — 直接取最大概率
- 二值模式: vis = (argmax后取one-hot最大值)
- 训练用连续, 测试可切换二值

### M3: Masked Pooling Head
- 每个部件用其概率掩码加权空间特征 → AdaptiveAvgPool
- 类似我们的 PosePartHead, 但掩码来自学习而非姿态

## 关键洞察
1. 学到的注意力掩码可能聚焦衣着而非骨骼 → 姿态引导更可靠
2. **融合思路**: ViTPose visibility (真实遮挡) ∩ BPBreID attention (学到的可见性)
3. GiLt Loss 在 KPR 和 BPBreID 中都用到, 验证了其有效性
