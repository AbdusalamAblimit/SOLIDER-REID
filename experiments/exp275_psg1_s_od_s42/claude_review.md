# Claude Review — exp275_psg1_s_od_s42

**审查对象**: Phase 3-A Small 1-stage PSG,对标 exp274 (Small no-PSG) + exp271 (Tiny 1-stage)

## 审查范围

1. `design.md` — 结构同 exp271(Tiny 1-stage),只换 backbone Small
2. 代码改动: 无(纯 CLI override,同 exp271/274 代码路径)
3. 与 exp274 单变量差异: `POSE_BACKBONE_PSG=False → True`, `POSE_PSG_STAGES=[-1]`
4. Phase 3-A Small 矩阵: exp274 → exp275 (本) → exp276 → exp277
5. PSG stage 3 注入路径已在 exp254a(Small 2-stage Full Scaffold)验证成熟

## 风险

- OOM: Small + 1-stage PSG 显存 ~7-8GB on Occ-Duke flip-eval,远低于 lab4090 24GB 上限,**无风险**
- 收敛: 和 exp271 同机制,Tiny 已 FINAL 60.2,Small 预期更高

## 结论

**审查通过**。
