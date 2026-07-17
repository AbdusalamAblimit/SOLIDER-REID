# 实验 exp191: OA-SD + CE (不用 SupCon)

## 动机
- exp188 测试 OA-SD + SupCon
- 需要消融：OA-SD 在 CE (不用 SupCon) 下是否也有效？
- 如果有效 → OA-SD 是独立有效的 paradigm
- 如果无效 → OA-SD 依赖 SupCon

## 技术方案
- 使用 exp166 配置（CE + per-token）+ PLBOA
- 增加 POSE_OA_SD: True
- 不使用 SupCon
- 远程 5060 Ti 16GB（2x forward 但 teacher 无 graph）

## 对照组
- exp166 (CE, 无 OA-SD): 63.1/73.9
- exp188 (SupCon + OA-SD): 运行中
