# 实验 exp084: Cross-Instance Pose-Guided Feature Recovery (CIPGFR)

## 动机
- exp083 PGFI (self-inpainting) 中性偏负 → 自我恢复没有足够的监督信号
- 但 feature recovery 范式本身是对的（FCFormer TPAMI 2024 SOTA）
- 关键区别：FCFormer 用 transformer decoder 做 self-recovery
- **我们的新想法**: 用同一 ID 的其他图片作为 recovery target — 跨图片监督

## 创新点
- **不是 self-recovery，是 cross-instance recovery**
- 对于同一 ID 的一对图片 (img_A, img_B):
  - 如果 A 的下半身被遮挡，B 的下半身可见
  - 用 B 的下半身 GCN 特征作为 A 的恢复目标
  - 训练 A 的 GCN branch 学会从上半身特征推断下半身
- 这利用了 batch 中的同 ID 正样本对（IdentitySampler 保证每个 ID 4 张图）

## 技术方案

### 在 GCN branch 上实现
```
For each pair (img_A, img_B) of same ID in batch:
  1. kp_feats_A: (17, 768) from GCN  # A 的 17 个关键点特征
  2. kp_feats_B: (17, 768) from GCN  # B 的 17 个关键点特征
  3. kp_weights_A: (17,) confidence   # A 的关键点可见性
  4. kp_weights_B: (17,) confidence   # B 的关键点可见性

  # 找到 A 遮挡但 B 可见的关键点
  5. recovery_mask = (kp_weights_A < threshold) & (kp_weights_B > threshold)

  # Recovery loss: A 的遮挡关键点特征应该接近 B 的可见关键点特征
  6. loss_recovery = MSE(kp_feats_A[recovery_mask], kp_feats_B[recovery_mask].detach())
```

### 关键设计
1. **Recovery target 是 detach 的** — 不让 recovery loss 影响 B 的 GCN 训练
2. **只在有互补的 pair 上计算** — 如果 A 和 B 遮挡的区域完全一样，跳过
3. **使用 GCN 输出的 kp_feats** — 经过图传播后的特征，不是原始采样特征
4. **Loss weight**: 从 0 开始 warmup，避免早期干扰 ID+Triplet 收敛

### 参数
- 零额外参数！只加一个 loss term
- 但与之前 5 个失败的辅助 loss 不同: 这个 loss 有**显式的跨图片监督目标**

### 与失败方向的区别
- vs SGMKC (exp048): SGMKC 做 self-reconstruction (mask then predict self)
  CIPGFR 做 cross-instance recovery (predict other view's features)
- vs CSGT (exp047): CSGT 改变 triplet mining 策略
  CIPGFR 加一个新的 recovery loss
- vs PAMC (exp050): PAMC 做 SimSiam-style consistency
  CIPGFR 做显式的 keypoint-level feature matching

## 对照
- exp066 PAA = 61.6%/74.2%
- exp030a 3-seed = 60.73%/72.57%

## 预期
- 如果有效: GCN branch 学到更好的 keypoint 表示 → fusion 增益增加
- 如果无效: 同 ID 不同视角的 keypoint 特征差异太大，MSE 不合适
