# 实验 exp030a: PSG + Skeleton GCN（无 PDS 架构）

## 动机
- exp030 (PDS+StopGrad+GCN) 达到 mAP 60.5% (concat_scaled), 但使用了 PDS 的独立 Stage 3 (+6.3M params)
- exp007a 证明 PDS 的增益完全来自 loss weighting（0.5x global loss）
- **核心问题**: GCN 的 +1.0% 增益是否需要 PDS 的独立 Stage 3？还是可以直接在 PSG backbone 上工作？
- 如果成功，整个方法简化为 PSG (+102K) + GCN (~400K) ≈ 0.5M params，vs PDS 的 6.3M

## 创新点 / 核心想法
- 将 Skeleton GCN 直接挂在 PoseBackboneModel 的输出上（共享同一个 Stage 3），不需要独立的 Part Stage 3
- GCN 输入使用 detached 的 backbone 特征（stop_grad），避免 GCN 梯度干扰 PSG backbone
- 返回 [global_score, gcn_score] 和 [global_feat, gcn_feat] 触发 list-loss 路径 → 自动获得 0.5x global loss 效果
- **不需要显式设置 GLOBAL_LOSS_SCALE**，因为 list-loss 已经隐式实现

## 技术方案

### 架构

```
Input Image + Pose Heatmaps
        ↓
 Swin Stage 0-2 (共享)
        ↓
 Stage 3 + PSG gates (共享，单一副本)
        ↓
   Feature Map (12×4, 768ch)
        ↓                ↓
      GAP          detach() → GCN Head
        ↓                ↓
  Global Feat      Skeleton Feat (768d)
        ↓                ↓
   BN + Cls         BN + Cls
        ↓                ↓
 [score0, score1]  [feat0, feat1]  → list-loss (w_g=0.5)
```

### 修改文件
1. `model/pose_backbone_model.py`: 添加 `POSE_SKELETON_GCN` 支持
   - `__init__`: 如果 `POSE_SKELETON_GCN=True`，创建 SkeletonGCNHead
   - `forward()`: 在 backbone 输出后，将 detached feature map 传给 GCN
   - 训练: 返回 [global_score, gcn_score], [global_feat, gcn_feat]
   - 测试: 根据 POSE_TEST_FEAT 决定返回 global/concat_scaled/equal_concat/gcn_only

2. `configs/occluded_duke/pose_psg_gcn.yml`: 新配置
   - POSE_BACKBONE_PSG: True (使用 PoseBackboneModel)
   - POSE_SKELETON_GCN: True (启用 GCN head)
   - 不设 GLOBAL_LOSS_SCALE（list-loss 已隐式 0.5x）
   - 不设 POSE_DUAL_STREAM（不使用 PDS）
   - POSE_TEST_FEAT: 'concat_scaled'

### 关键超参数
- GCN: 2 layers, hidden=256（与 exp030 相同）
- POSE_PART_WEIGHT: 1.0（w_g=0.5, w_p=0.5，与 PDS 相同）
- 其余所有超参数与 exp007 完全相同

### Stop Grad 设计
- GCN 输入使用 `featmap.detach()`：GCN 梯度不回流到 backbone
- 这等价于 PDS+StopGrad 中 Part 分支对共享层的隔离
- 好处: backbone 只被 global loss 的 0.5x 梯度优化 → 与 exp007a 一致

## 预期结果
- **如果假设成立** (GCN 不需要独立 Stage 3): mAP ~60-60.5%, 接近 exp030
- **如果 GCN 需要独立 Stage 3**: mAP ~59.5%（回退到 exp007a 水平，GCN 特征质量差）
- **关键指标**: global-only 应该 ≈ 59.5%（与 exp007a 一致，因为 GCN stop_grad）

## 对照组
- exp007a (PSG + 0.5x loss, 无 GCN): mAP 59.5%, R1 69.8%
- exp030-cs (PDS+SG+GCN, concat_scaled): mAP 60.5%, R1 70.5%
- 消融变量: 是否需要独立 Stage 3 给 GCN
