# 实验 exp036: Part-level Triplet Loss for GCN Branch

## 编号说明
- 按最初的 visibility 路线命名，`exp036` 原本预留给 visibility 后续阶段。
- 当前这个 `exp036` 实际上已经偏离原 visibility 路线，转而用于 `exp035` 之后的 GCN branch 内部探索。
- 后续在文档或论文中引用时，应把它表述为“`exp035` 之后的 branch 内部探索实验”，不能写成 visibility 路线的自然延续。

## 动机
- 当前 GCN 分支只有一个聚合后的 triplet loss，没有关键点级别的度量学习信号
- Phase 1 的 GiLt 实验（exp012）在更弱的 part feature 上实现了 +0.5% mAP
- 如果 GCN 的每个关键点特征都被显式训练为判别性的，则：
  1. 聚合特征（pooled feature）质量更高
  2. 可以为论文提供"关键点级判别性"的实验证据
  3. 为后续 Learnable Keypoint Attention 打基础

## 核心假设
- 对 GCN 分支的 17 个关键点特征施加独立 triplet loss，可以提升聚合特征的判别性
- 置信度加权确保遮挡关键点不产生噪声梯度

## 技术方案

### 修改文件
1. `model/modules/skeleton_gcn.py`: 修改 `SkeletonGCNHead.forward()` 在训练时额外返回 kp_feats_enhanced (B, 17, C)
2. `loss/triplet_loss.py`: 可能需要添加 per-keypoint triplet loss 计算
3. `processor/processor.py`: 在训练循环中计算 part-level triplet loss
4. `config/defaults.py`: 添加 `POSE_KP_TRIPLET` 开关和权重

### 数据流
1. GCN forward: 返回 skeleton_feat (B, C) + kp_feats_enhanced (B, 17, C) + kp_weights (B, 17)
2. 对每个关键点 k (0-16):
   - 提取 kp_feats[:, k, :] → (B, C)
   - 用 kp_weights[:, k] > threshold 的样本计算 triplet loss
   - 或用 kp_weights[:, k] 作为 triplet loss 的权重
3. part_tri_loss = weighted_mean(per_kp_triplet_losses)

### 关键超参数
- `POSE_KP_TRIPLET`: True/False 开关
- `POSE_KP_TRIPLET_WEIGHT`: loss 权重（初始 1.0，与现有 tri_part 权重相同）
- 不需要 BN / classifier per keypoint（太重，只需 triplet loss）

## 预期结果
- 乐观: +0.5~0.8% mAP（与 Phase 1 GiLt 一致或更好）
- 现实: +0.3~0.5% mAP
- 悲观: 0~0.2% mAP（GCN 消息传递已使特征判别性足够）
- 即使为 0 也有价值：证明 GCN 自动学习了关键点判别性

## 对照组
- Baseline: exp035a (PSG + GCN, score weight, equal_concat) = 61.1% mAP
- 消融变量: 仅增加 part-level triplet loss
