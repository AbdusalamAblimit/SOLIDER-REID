# 实验 exp032: PSG + Keypoint Pooling Only（无 GCN 图传播）

## 动机
- `exp030a` 中 `gcn_only` 和 `equal_concat` 表现很强，但 `exp030b` 表明低权重时图分支几乎没学好，仍然能得到较高的 `gcn_only` / `Rank-1`
- 这暴露了一个关键混杂因素：当前 head 不仅包含图卷积，还包含 **关键点采样 + 置信度加权池化**
- **核心问题**：真正有效的是 GCN 的骨架传播，还是 keypoint-guided pooling 本身？

## 核心想法
- 保持 `exp030a` 的所有训练设定不变
- 仍然使用 keypoint head 和 list-loss 路径
- 但在 head 内部关闭图传播，只保留：
  1. 在 17 个关键点位置从 Stage 3 featmap 做 bilinear sample
  2. 按关键点置信度做 weighted average

## 对照关系
- `exp007a`: PSG + 0.5x loss，无 keypoint head
- `exp030a`: PSG + keypoint pooling + GCN
- `exp032`: PSG + keypoint pooling only

## 关键配置
- 配置文件：`configs/occluded_duke/pose_psg_keypoint_pool.yml`
- 关键开关：
  - `POSE_SKELETON_GCN: True`
  - `POSE_KEYPOINT_POOL_ONLY: True`

## 预期解读
- 如果 `exp032 ≈ exp030a`：
  - 说明主要增益来自 keypoint pooling，GCN 传播贡献很小
- 如果 `exp032 << exp030a`：
  - 说明图传播确实提供了额外互补信息
- 如果 `exp032 > exp007a` 且 `gcn_only` 仍然强：
  - 说明 pose-guided sparse pooling 本身就是一个强基线
