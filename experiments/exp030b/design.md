# 实验 exp030b: PSG + GCN (Global Loss ≈ 1.0x)

## 动机
- exp030a 证明 PSG+GCN (equal_concat) 达到 mAP 61.1%
- 但 list-loss 隐式提供了 0.5x global loss (exp007a 证明这 +1.2%)
- **核心问题**: GCN 的 +1.3% (61.1% vs 59.8%) 增益中，有多少来自 GCN 特征本身，有多少来自隐式 loss scaling？
- 如果 PSG+GCN 在 1.0x global loss 下 global-only ≈ exp007 (58.3%)，则证明 GCN 不影响 global 特征质量
- 如果 equal_concat 仍 > global-only，则证明 GCN 特征本身提供互补信息

## 创新点 / 核心想法
- 将 POSE_PART_WEIGHT 设为极小值 (0.01)，使 w_g ≈ 0.99, w_p ≈ 0.01
- GCN 仍然训练（梯度来自 0.01 权重的 GCN loss），但 global loss 几乎不受影响
- 这是一个消融实验：隔离 GCN 特征贡献 vs loss scaling 贡献

## 技术方案

### 修改
- 仅 config 修改，无代码变更
- 新 config: `configs/occluded_duke/pose_psg_gcn_noscale.yml`
  - POSE_PART_WEIGHT: 0.01 (w_g=0.99, w_p=0.01)
  - 其余与 exp030a 完全相同

### 关键超参数
- POSE_PART_WEIGHT: 0.01 (核心变量)
- 其余同 exp030a

## 预期结果
- **global-only mAP ≈ 58.3%** (接近 exp007，因为 global loss ≈ 1.0x)
- **equal_concat mAP > 58.3%** (GCN 特征仍提供互补信息)
- 如果 equal_concat ≈ 59-60%，说明 GCN 特征本身贡献 ~1%
- 如果 equal_concat ≈ 58.3%，说明 GCN 特征无价值，所有增益来自 loss scaling

## 对照组
- exp007 (PSG, 1.0x loss, 无 GCN): mAP 58.3%, R1 67.9%
- exp030a-g (PSG+GCN, 隐式 0.5x, global): mAP 59.8%, R1 69.5%
- exp030a-eq (PSG+GCN, 隐式 0.5x, equal_concat): mAP 61.1%, R1 73.7%
- 消融变量: POSE_PART_WEIGHT 从 1.0 → 0.01 (loss scaling 效果被消除)
