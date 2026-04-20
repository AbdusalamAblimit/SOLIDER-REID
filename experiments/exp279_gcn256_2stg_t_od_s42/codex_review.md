# Codex Review — exp279_gcn256_2stg_t_od_s42

**Verdict**: approve
**Date**: 2026-04-20 22:05
**Review round**: 1

## Findings

零代码改动。相对 exp261 (Tiny Full Scaffold FINAL 65.9/77.4) 单变量改: `MODEL.POSE_GCN_HIDDEN` 512→256 (PSG_STAGES 保持默认 `[-2,-1]`)。

这是 Phase 3-B 矩阵里**最干净的单变量 ablation** — 回答"2-stage PSG 下 GCN cap 的边际收益"。exp279 vs exp278 = GCN256 基线上加 1 个 PSG stage 的边际贡献,是对 Phase 3-A 推测的 Small 版验证。

`POSE_GCN_HIDDEN=256` 直接引用 defaults (L140 的默认值),无 shape mismatch 风险。

## 结论

codex 审查通过。
