# Codex Review — exp283_gcn256_2stg_s_od_s42

**Verdict**: approve
**Date**: 2026-04-20 22:05
**Review round**: 1

## Findings

零代码改动。相对 exp262 (Small Full Scaffold FINAL 73.8/83.1) 单变量改: `MODEL.POSE_GCN_HIDDEN` 512→256 (PSG_STAGES 保持默认 `[-2,-1]`)。

**2-stage PSG 下 GCN cap 影响** (Small 版),直接对应 Tiny exp279。三角对照:
- exp283 vs exp282 = GCN256 基线上 PSG stage 数影响 (Small)
- exp283 vs exp262 = 2-stage PSG 下 GCN cap 影响 (Small)
- exp283 vs exp279 = 同构 scaffold 的 Tiny↔Small backbone 缩放

`POSE_GCN_HIDDEN=256` 引用 defaults,无 shape 风险。

## 结论

codex 审查通过。
