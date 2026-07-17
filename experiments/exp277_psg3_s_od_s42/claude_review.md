# Claude Review — exp277_psg3_s_od_s42

**审查对象**: Phase 3-A Small 3-stage PSG,对标 exp273 (Tiny 3-stage) + exp276 (Small 2-stage)

## 审查范围

1. `design.md` — 只改 `POSE_PSG_STAGES=[-3,-2,-1]`
2. 代码改动: 无
3. 与 exp276 单变量差异: PSG stages `[-2,-1]` → `[-3,-2,-1]`(加 stage 1)
4. Phase 3-A Small 矩阵收尾: exp274 → exp275 → exp276 → exp277(本)
5. PSG 三 stage 注入已在 exp273(Tiny 3-stage)同机制走通

## 风险

- OOM: Small + 3-stage PSG,额外显存开销 < 0.5GB,4090 24GB 充裕
- 收敛: 3-stage 浅层 PSG 可能拖累,历史 Tiny 3-stage ≤ Tiny 2-stage

## 结论

**审查通过**。
