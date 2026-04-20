# Codex Review — exp273_psg3_t_od_s42

**Verdict**: approve
**Date**: 2026-04-20 16:40
**Review round**: 1

## Findings

零代码改动。相对 exp272 只改 `MODEL.POSE_PSG_STAGES` 从 `[-2,-1]` → `[-3,-2,-1]`。PSG 三 stage 注入路径已在 exp254a (2-stage full scaffold) 全链路验证,代码成熟。CLI override 模式同 exp270/271/272,queue_on_ckpt.sh 的 `EXTRA_OVERRIDES` 机制在 exp271/272 已跑通。

## 结论

codex 审查通过。
