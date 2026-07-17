# Codex Review — exp249

**Verdict**: approve (after fixing OUTPUT_DIR)
**Date**: 2026-04-06 09:30
**Review round**: 1

## Findings

- **[high] OUTPUT_DIR 未在 design.md 的 CLI overrides 中列出。** base config 默认 OUTPUT_DIR 指向 exp244 目录，必须显式 override 为 `./log/occluded_duke/exp249_small_lgpa_gcn`。已在启动命令中添加。
- **[low] POSE_LGPA, POSE_LGPA_DETACH 已在 base yaml 中为 True，CLI override 冗余但无害。**
- **[low] TEST.IMS_PER_BATCH 128 偏保守 (exp245g 用 256 成功)，但安全。**
- **[low] claude_review.md 33 行，勉强过 30 行门槛。config-only 实验可接受。**

## 结论

codex 审查通过。唯一 actionable 是 OUTPUT_DIR 必须显式 override（已确认）。
