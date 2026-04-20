# Codex Review — exp284_gcn512_1stg_s_od_s42

**Verdict**: approve
**Date**: 2026-04-20 22:05
**Review round**: 1

## Findings

零代码改动。相对 exp262 (Small Full Scaffold FINAL 73.8/83.1) 单变量改: `MODEL.POSE_PSG_STAGES` `[-2,-1]`→`[-1]` (GCN_HIDDEN 保持默认 512)。

这是 **Phase 3-B 的 Small 核心最小闭环** (phase3_design.md L153) — 回答"Small + 高容量 GCN 下 1-stage vs 2-stage PSG"。

- exp284 vs exp262 = PSG stage 数独立影响 (Small 版)
- exp284 vs exp280 = 同结构 Tiny↔Small 缩放
- 四件套 (exp280/exp261/exp284/exp262) 构成 Phase 3-B 最小闭环: 2×2 = GCN512×PSG{1,2} × backbone{T,S}

CLI override `POSE_PSG_STAGES="[-1]"` 语法合法。

## 结论

codex 审查通过。
