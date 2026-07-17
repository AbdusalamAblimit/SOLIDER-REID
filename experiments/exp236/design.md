# 实验 exp236: FSDC on Tiny (正确增强配置)

## 动机
exp235 使用了错误的增强配置 (ROA=True + PLBOA prob=0.5)，与 exp191 baseline (ROA=False + PLBOA prob=0.7) 不一致。本实验修正配置，公平对比 FSDC。

## 技术方案
与 exp235 完全相同的 FSDC 配置，仅修正增强：
- `MODEL.POSE_ROA False` (关闭 ROA)
- `MODEL.POSE_LOWER_BODY_OCC_PROB 0.7` (与 exp191 一致)

## 对照组
- exp191 (ROA=False, PLBOA=0.7): 63.2/75.4
