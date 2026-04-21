# Claude Review — exp289_lgpaOnly_2stg_s_od_s42

**审查对象**: Phase 3-C Small 第 2 个 run, LGPA-only + 2-stage PSG

## 审查范围

- `design.md` — 相对 exp262 单变量 GCN True → False
- 代码改动: **无**
- daemon 76271 auto-chain after exp288

## 变量隔离

- 相对 exp262 单变量 GCN (PSG stages 保持 `[-2,-1]`)
- 相对 exp288 单变量 PSG stages (1 → 2-stg), LGPA-only scaffold 一致
- 相对 exp287 Tiny LGPA-only 2stg: backbone 缩放 Tiny → Small

## 核心问题

"Small 上 GCN 是否为 R1 关键?" — 通过 exp289 (no GCN + 2stg) vs exp262 (GCN512 + 2stg) 回答。

若 exp289 FINAL R1 ≥ 82.5: GCN 对 Small R1 也不是关键 (和 Tiny 一致, exp287 R1 -0.4 vs exp261)
若 exp289 FINAL R1 << 82: GCN 对 Small R1 必要

## OOM 风险

同 exp288, TEST.IMS_PER_BATCH 128 预防。

## 时间预算

同 exp288, ~10-13h on srvC 5060Ti。

## 结论

**审查通过**。Phase 3-C Small 2/2, 完成 4×4 LGPA-only 矩阵 (Tiny exp286/287 + Small exp288/289)。
