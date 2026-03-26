# 实验 exp194: OA-SD + CE + oa_sd_weight=2.0

## 动机
- exp191 (OA-SD + CE, weight=1.0) = 63.2/75.4 — 在 CE base (+2.9/+2.6)
- exp192 (OA-SD + CE, decay=0.99) = 62.6/74.9 — decay 不敏感
- 下一个需要验证的超参是 distillation loss 的权重
- 如果 weight=2.0 能进一步提升，说明 OA-SD 的 distillation 信号被低估了

## 核心假设
OA-SD distillation loss 当前权重 1.0 可能太弱，增大到 2.0 可以提供更强的 occlusion invariance 学习信号。

## 技术方案
- 配置 = exp191 + POSE_OA_SD_WEIGHT=2.0
- 即: 1-view + CE + PLBOA + OA-SD (EMA decay=0.999, weight=2.0)
- 无代码修改

## 预期结果
- 假设成立: mAP 63.5-64.0% (超过 exp191 的 63.2%)
- 如果失败: distillation 过强抑制 CE 的分类学习能力

## 对照组
- exp191 (OA-SD weight=1.0): 63.2/75.4
- exp166r (CE base): 60.3/72.8
