# 实验 exp105: SGRE (Skeleton-Guided Re-Encoding)

## 动机
- 为什么要做这个实验？
  - 当前 ReID 范式：独立编码 → cosine 距离。此范式忽略了 pair-specific 遮挡信息
  - SGCFR (+2.6%) 证明：**最优表示取决于 query-gallery pair 的共同可见结构**
  - 但 SGCFR 是手工规则（可见关键点加权平均），不是学习的
  - **SGRE 用 cross-attention 学习最优 pair-conditioned 比较策略**
- 基于哪些前序实验？
  - SGCFR: 证明 pair-specific 推理有巨大价值 (+2.6%)
  - exp099 OT: 证明简单 OT 不够 (-2.6%)，需要更强的比较机制
  - GCN branch: 已有 per-keypoint 特征 (17×768)，可直接用于 cross-attention

## 创新点 / 核心想法
- 核心假设：
  **用 cross-attention 在 query 和 gallery 的关键点特征之间做推理，可以学到比固定 cosine/visibility-weighted 更好的比较策略**
- 与 baseline 相比：
  - 新增 SkeletonReEncoder 模块 (~200K params)
  - 新增 SGRE triplet loss (on detached kp_feats)
  - 测试时：global cosine 初筛 → SGRE re-rank top-100

## 技术方案
- 修改文件：
  - `model/modules/skeleton_reencoder.py`: 新模块 (cross-attention + sim head)
  - `model/pose_backbone_model.py`: 注册 SGRE 模块
  - `processor/processor.py`: SGRE triplet loss
  - `scripts/eval_sgre.py`: 测试时 re-ranking
- 数据流：
  1. GCN branch → kp_feats (B, 17, 768) + kp_weights (B, 17)
  2. SGRE 训练: 对 batch 内的 (anchor, pos, neg) 三元组，
     计算 SGRE similarity，triplet loss
  3. SGRE 测试: 对每个 query 的 top-100 gallery 候选，
     计算 SGRE similarity，重新排序
- 关键超参数：
  - d_model = 256, nhead = 4, num_layers = 2
  - SGRE loss weight = 0.5
  - Re-rank top-K = 100

## 预期结果
- 训练: SGRE loss 教 GCN 产出更适合 pair comparison 的 kp 特征
- 测试: re-rank 应该提升 mAP +1~3%（类似 SGCFR 但学习的）
- 如果失败：可能是 200K params 在 15K 数据上过拟合

## 对照组
- exp066 baseline: 61.6%/74.2%
- exp066 + SGCFR: 64.2%/75.7% (test-time recovery)
- SGRE 目标: 超越 SGCFR 的 test-time 效果
