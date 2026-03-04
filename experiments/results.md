# 实验结果总表

## 数据集: Occluded-Duke

| ID | 方法 | mAP | R-1 | R-5 | R-10 | 备注 |
|----|------|-----|-----|-----|------|------|
| 001 | Baseline (SOLIDER-Swin-Tiny) | — | — | — | — | 纯 baseline |
| 002 | VPReID v1 (global+parts+fg) | — | — | — | — | 完整 VPReID |
| 003 | VPReID w/o Part ID Loss | — | — | — | — | 消融: PART_ID_WEIGHT=0 |
| 004 | VPReID w/o Push Loss | — | — | — | — | 消融: PUSH_WEIGHT=0 |

## 评估模式说明
- global: 仅用全局特征
- parts: 拼接5个部件特征
- fused: 0.5*global + 0.5*parts 距离

（待实验结果填充）
