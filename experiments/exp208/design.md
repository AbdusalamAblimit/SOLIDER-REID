# 实验 exp208: Small + GCN+PAA+CE+OA-SD + 0.5x Global Loss

## 动机
- exp206 (Small GCN+PAA+CE+OA-SD, GLOBAL_LOSS_SCALE=1.0) = 70.5/82.3
- 0.5x global loss 在 Tiny 上已确认 +1.53% mAP (exp007a vs exp007)
- **但从未在 Small 上测试过！** 所有 Small/Base 实验都用 1.0
- 如果 0.5x 在 Small 上也有效: 70.5 + 1.5 = **72.0%**

## 核心假设
0.5x global loss scale 的增益在 Small 上与 Tiny 类似 (+1-2%)。

## 技术方案
- 配置 = exp206 + MODEL.GLOBAL_LOSS_SCALE 0.5
- 无代码修改

## 对照组
- exp206 (GLOBAL_LOSS_SCALE=1.0): 70.5/82.3
