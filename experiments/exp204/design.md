# 实验 exp204: Tiny + SupCon + PLBOA + ROA (远程, 1-view)

## 动机
- PLBOA: pose-guided lower-body occlusion (自生成遮挡)
- ROA: realistic occlusion augmentation (VOC 真实物体 paste)
- 目前 SupCon 实验只用 PLBOA，没有同时用 ROA
- 4090 PAA 配置用了 ROA（无 PLBOA），效果好
- **测试 PLBOA+ROA 双重遮挡增强是否叠加**

## 核心假设
PLBOA 和 ROA 是正交的遮挡类型（PLBOA=下半身遮挡，ROA=随机物体遮挡），同时使用应提供更多样的训练信号。

## 技术方案
- 在 exp176 (SupCon T=0.05, 1-view) 基础上加 ROA
- 配置: POSE_ROA=True + POSE_LOWER_BODY_OCC=True

## 对照组
- exp176 (SupCon, PLBOA only): 64.1/75.5
