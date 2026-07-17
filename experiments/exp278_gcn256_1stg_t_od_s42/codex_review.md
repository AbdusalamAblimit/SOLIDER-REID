# Codex Review — exp278_gcn256_1stg_t_od_s42

**Verdict**: approve
**Date**: 2026-04-20 22:05
**Review round**: 1

## Findings

零代码改动。Phase 3-B 首个 run,相对 Phase 1 exp261 (Tiny Full Scaffold FINAL 65.9/77.4) 两变量同时改:
- `MODEL.POSE_GCN_HIDDEN` 512→256
- `MODEL.POSE_PSG_STAGES` `[-2,-1]`→`[-1]`

变量隔离说明在 design.md 中明确,exp278 vs exp280 = "GCN cap 差异"对照,exp278 vs exp279 = "PSG stage 差异"对照。双变量同改的结果对标 exp261 作为 2x2 矩阵一角。

`POSE_GCN_HIDDEN=256` 在 config/defaults.py (L140) 是合法默认值,`POSE_PSG_STAGES="[-1]"` 在 Phase 3-A exp271 已验证。

CLI override 语法与 Phase 3-A 规范一致,yacs 能正确解析 list 字面量。

## 结论

codex 审查通过。
