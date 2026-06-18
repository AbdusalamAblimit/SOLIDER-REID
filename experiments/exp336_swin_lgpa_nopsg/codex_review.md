# Codex Review — exp336 Swin 纯 LGPA-D 隔离

**Verdict: approve**

## Checks passed
- `POSE_PSG_STAGES: []` 真禁 PSG:psg_stage_indices/psg_modules_dict 空、backward-compat 跳过、`_run_stage_with_psg` 任何 stage 都不触发。
- `POSE_BACKBONE_PSG: True` 仅在 make_model:467 选 PoseBackboneModel,不强制 PSG。
- OA_SD/PARALLEL_AUG/LOWER_BODY_OCC=False 干净跳过 dataloader/processor 分支;单视图 collate 正常(pose_dataset:1089)、processor 走标准 tensor 路径(processor:547)。
- 不设 POSE_USE_TARGET_HEATMAP → 默认 False → LGPA 收 scene-merged 热图;`lgpa_assign~7` 是对的 sanity。
- **eval 协议有效**:同一 ckpt 只 override `MODEL.POSE_TEST_FEAT`(equal_concat vs global)。**关键:global 测试别设 POSE_LGPA=False**(会改 ckpt 加载/架构 parity)——只改 POSE_TEST_FEAT。

## Low（已知,不阻断）
- design.md 把 `global == 无-LGPA baseline` 说过了:严格说是"同-ckpt global-only",非独立训练的 no-LGPA baseline(GLOBAL_LOSS_SCALE 0.5 + list-loss 改了 global 梯度尺度)。**within-checkpoint ablation 有效**("detached LGPA 描述子是否给该 ckpt 的 global 加值"),措辞改为"同-ckpt global-only"即可。已在 claude_review 注明。

## 结论
codex 审查通过。训练命令注意:test 阶段只 override POSE_TEST_FEAT,不动 POSE_LGPA。
