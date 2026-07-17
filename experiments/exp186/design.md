# 实验 exp186: SupCon T=0.05 WITHOUT PSG (no backbone pose injection)

## 动机
- 当前所有 SupCon 实验都带 PSG backbone injection
- PSG 在 CE 环境下 +1.7% mAP
- 问题：PSG 在 SupCon 环境下贡献多大？
- 如果 PSG 贡献小 → SupCon 减少了对 PSG 的依赖
- 如果 PSG 贡献大 → PSG 仍是重要组件

## 技术方案
- 基于 exp176 最佳配置但禁用 PSG gating
- **不能用 POSE_BACKBONE_PSG=False**（会切换到错误的模型类 PoseReIDModel，丢失 STD-PR/PAPE/SupCon）
- 正确方法: `POSE_PSG_STAGES '[]'` — 保持 PoseBackboneModel 但不创建任何 PSG modules
- 所有 stage 走 normal path（无 pose gating）
- STD-PR + per-token + PLBOA + SupCon T=0.05 仍然生效（只去掉 PSG gating）

## 对照组
- exp176 (with PSG, SupCon): 64.1/75.5
- Baseline without PSG (CE): ~56.6/66.5 (original baseline)
