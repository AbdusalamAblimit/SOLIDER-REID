# Codex Review — exp286_lgpaOnly_1stg_t_od_s42

**Verdict**: approve
**Date**: 2026-04-20 23:30
**Review round**: 1

## Findings

零代码改动。Phase 3-C 首个 run,双变量组合相对 exp261 (Tiny Full Scaffold 65.9/77.4):
- `MODEL.POSE_SKELETON_GCN` True → False
- `MODEL.POSE_PSG_STAGES` `[-2,-1]` → `[-1]`

两变量均在 Phase 3-A/3-B 已充分验证,yacs 解析 + scaffold yml 继承路径已跑过多次。与 exp280 (GCN512 + 1-stage) 形成 Phase 3-C vs 3-B 的 "GCN on/off" pair。

srvC 上 Occ-Duke 4.9GB 数据 + pretrained 全齐,无数据缺失风险。

## 结论

codex 审查通过。
