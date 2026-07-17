# 实验 exp192: OA-SD + CE with faster EMA (decay=0.99)

## 动机
- exp191 (OA-SD + CE, decay=0.999): 63.2/75.4 — 独立有效
- 问题：EMA decay 0.999 是否最优？更快更新（0.99）可能让 teacher 更紧跟 student
- 消融 decay 参数

## 对照组
- exp191 (decay=0.999): 63.2/75.4
