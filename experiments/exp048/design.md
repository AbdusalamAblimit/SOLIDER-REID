# 实验 exp048: Skeleton-Guided Masked Keypoint Completion (SGMKC)

## 动机
- 47 个实验中所有在 PSG+GCN 基础上的架构添加和 loss 变体都失败了
- 但尚未尝试过 **self-supervised 训练策略** 改进 GCN branch
- FCFormer 用 transformer decoder 做 feature completion，PersonMAE 用 random patch masking
- **没有人在 skeleton graph 上做过 masked keypoint prediction for ReID**
- 如果 GCN 能学会从可见关键点恢复遮挡关键点，其特征应更具遮挡鲁棒性

## 创新点 / 核心假设
**假设**: 在 GCN 训练时随机 mask 关键点特征并强制重建，可以教会 GCN 利用骨架拓扑进行特征补全，从而产生更鲁棒的关键点表征。

与 baseline (exp030a) 相比，只改了一个变量：GCN 训练时增加 masked prediction auxiliary task。

## 技术方案

### 修改文件
1. `model/modules/skeleton_gcn.py` — `SkeletonGCNHead.forward()` 增加 masking 逻辑
2. `processor/processor.py` — 增加 reconstruction loss 计算
3. `config/defaults.py` — 增加 SGMKC 相关默认值
4. `configs/occluded_duke/exp048_sgmkc.yml` — 新实验配置

### 数据流
```
训练时:
  keypoint features (B, 17, 768)
        ↓
  random mask 30% keypoints → masked_features (B, 17, 768) [部分为零]
        ↓ [同时保存 original features]
  GCN propagation along skeleton edges
        ↓
  enhanced features (B, 17, 768)
        ↓
  ├── confidence-weighted average → skeleton_feat → ID/Triplet loss (主 loss)
  └── MSE at masked positions vs original features → recon_loss (辅助 loss)

测试时:
  keypoint features (B, 17, 768) → GCN → skeleton_feat [无 masking]
```

### 关键超参数
- `POSE_SGMKC = True`
- `POSE_SGMKC_RATIO = 0.3` (30% keypoints masked)
- `POSE_SGMKC_WEIGHT = 1.0` (reconstruction loss weight)
- 其余与 exp030a 完全相同

### masking 策略
- 每个样本独立生成随机 mask (B, 17)，每个关键点以 p=0.3 概率被 mask
- mask 后特征设为 0（类似 dropout）
- reconstruction target 是 mask 前的原始特征（detach，stop gradient）

## 预期结果
- **理想**: equal_concat mAP > 61.5% (+0.8% over exp030a mean) — 说明 masked training 改善了 GCN 特征
- **中性**: equal_concat mAP ≈ 60.7% — 说明 masked training 既不帮助也不伤害
- **负面**: equal_concat mAP < 60.0% — masking 破坏了 GCN 的主任务学习
- 失败最可能原因：GCN 的 2-layer 容量不足以同时完成 ID 分类和 feature completion

## 对照组
- Baseline 对照：exp030a (PSG+GCN, equal_concat, 3-seed mean = 60.73%)
- 消融变量：仅增加了 SGMKC auxiliary task，其余完全相同

## 止损规则
- 只跑一次（seed 1234），不做变体
- 如果 epoch 40 时 concat_scaled mAP 明显低于 exp030a 同期，考虑提前终止
