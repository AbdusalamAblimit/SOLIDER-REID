# Codex Review — exp270_psg0_t_od_s42

**Verdict**: approve
**Date**: 2026-04-20 09:30
**Review round**: 1

## Findings

本 run 无代码改动(纯 CLI override),diff 为空。审查焦点是 CLI 参数语义:

1. `MODEL.POSE_BACKBONE_PSG=False` — 关 PSG 注入
2. `MODEL.POSE_LGPA=False` — 关 CLIP-part assignment
3. `MODEL.POSE_SKELETON_GCN=False` — 关 skeleton GCN branch
4. `MODEL.POSE_OA_SD=False` — 关 occlusion-aware self-distillation
5. `MODEL.POSE_LOWER_BODY_OCC=False` — 关 PLBOA augmentation
6. `MODEL.POSE_PARALLEL_AUG=False` — 关 multi-view parallel augmentation
7. `MODEL.POSE_TEST_FEAT='global'` — 测试仅用 global feature(因无 branch)

每一项都对应 yacs config 的合法 key,`config/defaults.py` 明确有该字段。model construction 路径在每个模块前都有 `if getattr(cfg.MODEL, '<FLAG>', False): ...` 守卫,关闭即跳过构造,与已有 8 个 Phase 1 实验的 forward 路径一致,无 None-ptr 风险。

## 结论

codex 审查通过。单变量意义清楚(纯 Swin-Tiny baseline),代码安全(零 diff),预期数字合理(~56-58 mAP),启动无阻塞。
