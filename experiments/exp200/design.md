# 实验 exp200: OA-RD + CE (远程对照)

## 动机
- exp199 (OA-RD + SupCon + 3-view) 正在本地跑
- 需要对照：OA-RD 在 CE-only 路线下是否也有效？
- 与 exp191 (OA-SD + CE) 对比：relational distillation vs feature distillation

## 核心假设
OA-RD (relational distillation) 在 CE 路线下应与 OA-SD 效果相当或更好。

## 技术方案
- 配置 = base arch + CE + PLBOA + OA-RD (temp=0.1, weight=1.0)
- 无代码修改（复用 exp199 的 OA-RD 代码）
- 远程 16GB: 1-view + OA-RD

## 对照组
- exp191 (OA-SD + CE): 63.2/75.4
- exp166r (CE base): 60.3/72.8
