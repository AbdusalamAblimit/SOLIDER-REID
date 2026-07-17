# Codex Review — exp282_gcn256_1stg_s_od_s42

**Verdict**: approve
**Date**: 2026-04-20 22:05
**Review round**: 1

## Findings

零代码改动。相对 exp262 (Small Full Scaffold FINAL 73.8/83.1) 两变量同时改:
- `MODEL.POSE_GCN_HIDDEN` 512→256
- `MODEL.POSE_PSG_STAGES` `[-2,-1]`→`[-1]`

Phase 3-B Small 系列首个 run,对应 Tiny 的 exp278 在 Small backbone 上的复刻。

与 Phase 3-A exp275 (Small no-scaffold 1-stage PSG) 不同: exp282 开了 full scaffold (LGPA + GCN + OA-SD + ParAug + LOWER_BODY_OCC)。两者 Δ = "Full scaffold vs pure PSG" 差。

CLI override 与 exp278 Tiny 版本一致,不同仅为 config path (prcv_best_small.yml) 和 OUTPUT_DIR。

## 结论

codex 审查通过。
