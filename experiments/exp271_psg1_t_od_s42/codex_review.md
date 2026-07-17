# Codex Review — exp271_psg1_t_od_s42

**Verdict**: approve
**Date**: 2026-04-20 11:58
**Review round**: 1

## Findings

本 run 无代码改动,diff 为空。CLI 参数相对 exp270 只多两个:

1. `MODEL.POSE_BACKBONE_PSG=True` — 启用 PSG gate(Phase 1 默认)
2. `MODEL.POSE_PSG_STAGES="[-1]"` — 仅 stage 3 注入(yacs list,本 codebase 约定)

这两个键在 `config/defaults.py` 有定义,`PoseBackboneModel` 构造路径走 `make_model.py:467` 的 `elif` 分支,与 Phase 1 九个 exp 一致。`POSE_PSG_STAGES=[-1]` 等同于 exp007 历史配置(第 一次 PSG 验证,58.3/67.9 3-seed mean)。

其他 5 个 pose 模块 flag 均 False,与 exp270 保持一致,保证 Phase 3-A 矩阵内单变量 = PSG stage 数。

## 结论

codex 审查通过。单变量 PSG ablation 逻辑清楚,无代码修改,预期数字落在 exp007 历史值 +default flip bonus ~59-60,可启动。
