# 实验 exp198: STM + OA-SD + CE (远程)

## 动机
- exp197 (STM + SupCon + 3-view) 正在本地跑
- 远程 16GB 无法跑 3-view，用 1-view + OA-SD + CE + STM
- 测试 STM 在 OA-SD 路线下的效果（与 exp191 对照）

## 核心假设
STM（token mixup）和 OA-SD（self-distillation）是正交的增强手段，应能叠加。

## 技术方案
- 配置 = exp191 (OA-SD + CE) + POSE_STM=True
- 无代码修改（复用 exp197 的 STM 代码）
- 远程 16GB: 1-view + OA-SD + STM

## 对照组
- exp191 (OA-SD + CE, no STM): 63.2/75.4
