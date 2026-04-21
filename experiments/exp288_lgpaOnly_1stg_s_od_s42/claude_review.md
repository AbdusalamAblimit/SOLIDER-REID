# Claude Review — exp288_lgpaOnly_1stg_s_od_s42

**审查对象**: Phase 3-C Small 第一个 run, LGPA-only + 1-stage PSG

## 审查范围

1. `design.md` — 双变量组合 (GCN False + 1-stg PSG) 相对 exp262 的隔离
2. 代码改动: **无**
3. srvC 刚完成 exp287 (Phase 3-C Tiny 2-stg, 65.9/77.0), GPU 空闲 → 立即接
4. CLI TEST.IMS_PER_BATCH 128 (5060Ti Small Full eval 预防 OOM)

## 变量隔离

- 相对 exp262 (Small Full Scaffold GCN512+2stg FINAL 73.8/83.1) 两变量同改:
  - GCN 关 (同 Phase 3-C Tiny)
  - PSG stage 1 (同 exp288 设计)
- 对照 exp282 (Full GCN256+1stg 73.7/83.9): exp288 相对 exp282 单变量 (GCN False)
- 对照 exp287 (Tiny LGPA-only 2stg 65.9/77.0): exp288 是 Small 版的 1stg 对照
- 对照 exp286 (Tiny LGPA-only 1stg 66.0/76.6): exp288 是 Small 缩放

## OOM 风险

- Small Full-GCN 比 Full Scaffold 略轻 (少 GCN 512 head 参数)
- eval 峰值估计 ~12-13GB @ TEST 256
- **TEST.IMS_PER_BATCH 128 降低 eval 峰值 ~6-7GB**, 5060Ti 16G 安全
- 训练 IMS_PER_BATCH 64 保持

## 时间预算

- exp286 Tiny LGPA-only 305s/epoch × 120 = 10h
- Small LGPA-only 应该同 Tiny (LGPA-only 相对 Full 没那么重), srvC 5060Ti ~300-400s/epoch
- 总训练 10-13h → FINAL tmr 07-10 CST

## 预期

若 exp288 FINAL ≥ 73.5/83.0: LGPA-only Small 接近 Full Scaffold (类似 Tiny 模式)
- 支持论文: "semantic branch 已提供主要增益, GCN 仅 R1 锦上添花"
若 exp288 FINAL << 73.5: GCN 在 Small 上有实质贡献, 需进一步分析

## 结论

**审查通过**。零代码改动, Phase 3-C Small 首 run, 完成 semantic branch 依赖性消融的 Small 版。
