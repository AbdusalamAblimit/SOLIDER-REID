# exp036: Per-Keypoint Triplet Loss for GCN Branch — 监控日志

## 实验概述
- **目的**: 对 GCN 分支 17 个关键点特征施加独立 triplet loss
- **Base**: exp035a (PSG + GCN, score weight, equal_concat) = 61.1% mAP / 73.8% R1
- **变量**: 仅增加 POSE_KP_TRIPLET=True, POSE_KP_TRIPLET_WEIGHT=1.0
- **PID**: 3244784

## [06:39] 检查点 #1
**状态**: 🟡关注
**进度**: Epoch 3/120

| 指标 | 当前值 | 备注 |
|------|--------|------|
| Total Loss | 21.3 (E3 iter20) | E1 全 inf，E2 恢复 |
| id_global | 6.537 | 正常 |
| id_part | 6.429 | 正常 |
| tri_global | 2.160 | 正常 |
| tri_part | 4.217 | 正常 |
| tri_kp | 11.628 | E1=inf, E2=16.9→13.6, E3=11.6 |
| Acc | 0.014 | 正常起步 |
| LR | 1.27e-04 | warmup 中 |

**观察**: tri_kp 在 epoch 1 全程 inf（SoftMarginLoss 溢出），epoch 2 恢复到有限值。AMP scaler 应对了 inf gradients。其他 loss 分量正常。tri_kp 数值远大于 tri_part（11.6 vs 4.2），kp_triplet 主导 GCN 梯度。
**决策**: 继续观察，如果 epoch 10 eval 正常则无需干预
