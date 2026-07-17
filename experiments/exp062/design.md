# 实验 exp062: Learned Keypoint Uncertainty (LKU)

## 动机
- 当前 GCN 的 keypoint 特征加权使用 ViTPose confidence scores（固定、外部）
- 但 confidence 只反映 pose 估计的准确性，不反映特征对 ID 判别的可靠性
- 例如：一个关键点可能 pose 准确（高 confidence）但处于遮挡边缘（低 ID 可靠性）
- **LKU 的想法**: 训练一个轻量 MLP 从 GCN 特征本身预测每个关键点的 uncertainty，然后在 test-time 距离计算中用 uncertainty 加权

## 创新点 / 核心想法
- **核心假设**: 从 GCN 特征学出的 uncertainty 比外部 pose confidence 更能反映 ID 判别的可靠性
- **与 CVK 的区别**: CVK 用固定 confidence 做 binary common-visible filtering；LKU 用学习的 continuous uncertainty 做 soft weighting
- **训练方式**: uncertainty head 参与 ID loss 和 triplet loss 的计算——高 uncertainty 的关键点在 loss 中被降权
- **论文叙事**: "Pose estimation confidence ≠ identity-discriminative reliability. We learn per-keypoint uncertainty from features, enabling adaptive distance computation."

## 技术方案
- 在 SkeletonGCNHead 中加一个 `uncertainty_head`: Linear(768, 1) → sigmoid → (B, 17)
- 训练时: 用 uncertainty 加权 triplet loss 中的关键点距离
  - kp_dist(i) = ||kp_feat_q(i) - kp_feat_g(i)||^2
  - weighted_dist = sum(kp_dist(i) * (1 - uncertainty(i))) / sum(1 - uncertainty(i))
- 测试时: 在 equal_concat 模式中，用 learned uncertainty 而非 confidence 加权
- 额外正则: 对 uncertainty 加 mean penalty 防止 collapse（所有 uncertainty 趋向 1）

## 预期结果
- 如果成功: learned uncertainty 产生更好的距离度量 → mAP/R1 提升
- 如果失败: uncertainty 退化为 uniform（所有 keypoint 等权）或 collapse
- 需要多 seed 验证

## 对照组
- Baseline: exp030a (PSG+GCN, kp_weight=confidence)
- 消融变量: 用 learned uncertainty 替代 confidence weighting
