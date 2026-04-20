# Claude Review — exp276_psg2_s_od_s42

**审查对象**: Phase 3-A Small 2-stage PSG,对标 exp272 (Tiny 2-stage) + exp275 (Small 1-stage)

## 审查范围

1. `design.md` — 只改 `POSE_PSG_STAGES=[-2,-1]`,其他同 exp275
2. 代码改动: 无(纯 CLI override)
3. 与 exp275 单变量差异: PSG stages `[-1]` → `[-2,-1]`
4. 2-stage PSG 已在 exp272(Tiny 2-stage 进行中)和 exp254a(Small 2-stage Full Scaffold FINAL 74.0/84.0)验证

## 风险

- OOM: Small + 2-stage PSG,峰值 ~9GB,4090 24GB 充裕
- 收敛: 机制成熟

## 结论

**审查通过**。
