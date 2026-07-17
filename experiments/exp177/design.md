# 实验 exp177: SupCon + PAPE (无 multi-stage PSG)

## 动机
- exp174 SupCon + triple injection = 63.6/75.3 (突破)
- 问题：SupCon 的增益有多少来自 triple injection vs SupCon 本身？
- 本实验测试 SupCon + PAPE + PSG@Stage3 only（exp171 架构 + SupCon）

## 技术方案
- 基于 exp171 配置 (PAPE + PSG@Stage3 only)
- 增加 POSE_STR_SUPCON: True
- 不使用 multi-stage PSG

## 对照组
- exp171 (PAPE + CE): 63.2/74.3
- exp174 (PAPE + triple + SupCon): 63.6/75.3
- 消融：isolate SupCon on simpler architecture
