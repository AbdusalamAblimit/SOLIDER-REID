# Codex Review — exp280_gcn512_1stg_t_od_s42

**Verdict**: approve
**Date**: 2026-04-20 22:05
**Review round**: 1

## Findings

零代码改动。相对 exp261 (Tiny Full Scaffold FINAL 65.9/77.4) 单变量改: `MODEL.POSE_PSG_STAGES` `[-2,-1]`→`[-1]` (GCN_HIDDEN 保持默认 512)。

这是 **Phase 3-B 的 Tiny 核心最小闭环** (phase3_design.md L153) — 回答"高容量 GCN + 1-stage vs 2-stage PSG 谁更强"。

- exp280 vs exp261 = PSG stage 数的独立影响,GCN 容量相同
- 与 Phase 3-A exp271 (Tiny no-scaffold 1-stage 60.2/69.5) 对比 → scaffold 加成贡献量化

CLI override `POSE_PSG_STAGES="[-1]"` 与 Phase 3-A exp271 一致,yacs 解析无问题。

## 结论

codex 审查通过。
