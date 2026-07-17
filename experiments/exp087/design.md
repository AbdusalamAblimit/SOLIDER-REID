# 实验 exp087: Momentum Memory Contrastive Learning

## 动机
- DPEFormer 的核心 loss: HybridMemory 对比学习
- 当前 ID+Triplet 只在 batch 内做 mining（64 samples, 16 IDs）
- Momentum Memory 维护全类别特征记忆库，每次训练看到所有 702 个 ID 的特征
- 更多负样本 → 更好的判别性

## 技术方案
- MomentumMemory: per-class feature bank (702, 768)
- 每步: 1) 用当前 batch 更新 memory (EMA) 2) 对 memory bank 计算 contrastive CE loss
- Loss = ID + Triplet + mm_weight * Memory_Contrastive

## 参数
- feat_dim=768, num_classes=702, momentum=0.1, temp=0.05, weight=0.5
- 零额外可训练参数（memory 是 buffer）

## 对照
- exp066 PAA = 61.6%/74.2%
