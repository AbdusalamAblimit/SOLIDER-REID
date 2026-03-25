# 实验 exp181: SupCon T=0.05 without PLBOA

## 动机
- SupCon 在所有架构上都有正效果（+1.0~1.6 R1）
- 但当前所有 SupCon 实验都带 PLBOA
- 需要 isolate SupCon 效果：无 PLBOA 时 SupCon 是否仍有效？
- 这是论文消融的关键实验

## 技术方案
- 基于 exp176 配置但禁用 PLBOA
- Triple injection + SupCon T=0.05, POSE_LOWER_BODY_OCC: False

## 对照组
- exp166r (CE, no PLBOA): 60.3/72.8
- exp176 (SupCon T=0.05 + PLBOA): 64.1/75.5
