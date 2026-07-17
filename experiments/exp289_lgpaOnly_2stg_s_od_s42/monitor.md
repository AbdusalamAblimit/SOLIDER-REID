# exp289 monitor — Phase 3-C Small LGPA-only 2-stg PSG (Occ-Duke, seed 42)

- 机器: srvC (5060Ti 16G, TEST.IMS_PER_BATCH 128)
- 启动: 2026-04-22 12:52 CST (auto-chain from exp288 FINAL via daemon)
- Log: `/hy-tmp/log/occluded_duke/exp289_lgpaOnly_2stg_s_od_s42/train_log.txt`
- Config: `configs/occluded_duke/prcv_best_small.yml` + CLI `SEED 42 MODEL.POSE_SKELETON_GCN False TEST.IMS_PER_BATCH 128`
- Scaffold: Swin-Small + LGPA + OA-SD + ParAug + LOWER_BODY_OCC + **2-stage PSG [-2,-1]** (**无 GCN**)
- Speed: ~8 min/epoch × 120 = 16h 48min (实际 16h48min 12:52 → 05:39)

## 训练轨迹 (flip-test, eq_concat global)

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 55.4 | 66.7 | 81.1 | 86.4 |
| 20 | 63.7 | 74.5 | 85.6 | 88.4 |
| 30 | 68.9 | 79.3 | 88.6 | 91.2 |
| 40 | 69.3 | 79.1 | 88.1 | 90.7 |
| 50 | 72.1 | 82.3 | 89.6 | 91.5 |
| 60 | 72.0 | 81.9 | 90.0 | 91.9 |
| 70 | 73.0 | 83.3 | 90.5 | 92.8 |
| 80 | 73.4 | 83.4 | 90.3 | 92.6 |
| 90 | 73.6 | 83.2 | 90.2 | 92.3 |
| 100 | 73.8 | 83.5 | 90.4 | 92.5 |
| 110 | 73.8 | 83.2 | 90.4 | 92.5 |
| **120 FINAL** | **73.8** | **83.3** | **90.5** | **92.4** |

## FINAL (2026-04-23 05:39:56 CST)

- **mAP: 73.8%**, **Rank-1: 83.3%**, R5: 90.5%, R10: 92.4%
- 🔥 **对照 exp288 LGPA-only 1-stg FINAL**: 73.8/83.8/90.5/92.0 → Δ **0 / -0.5 / 0 / +0.4**
- **对照 exp285b Full Scaffold**: 73.8/83.8/90.7/92.7 → Δ **0 / -0.5 / -0.2 / -0.3**
- **对照 exp282 Full GCN256+1stg**: 73.7/83.9/90.5/92.5 → Δ +0.1/-0.6/0/-0.1
- Ckpt: `transformer_120.pth` (~222MB)

## 🎯 Phase 3-C Small 完整闭合 (2×2)

| | PSG `[-1]` | PSG `[-2,-1]` |
|---|-----------|----------------|
| **LGPA-only (无 GCN)** | **exp288 73.8/83.8** | **exp289 73.8/83.3** |

**核心结论**:
1. **mAP 完全持平**: 两 variants = 73.8 (= Full Scaffold exp285b 73.8) → GCN 零贡献
2. **R1 1-stg 优**: exp288 (1-stg) 83.8 > exp289 (2-stg) 83.3 → 和 Phase 3-B Full Scaffold Small 2×2 相反 (那里 2-stg 略优)
3. **R10 2-stg 优** (exp289 92.4 > exp288 92.0): 2-stg PSG 对 deep CMC 有正贡献

## 🎯 Phase 3-C Tiny+Small 跨 backbone 一致

| Backbone | 1-stg | 2-stg | 差异 |
|----------|-------|-------|------|
| Tiny (exp286/287) | 66.0/76.6 | 65.9/77.0 | 2-stg R1 微优 +0.4 |
| **Small (exp288/289)** | 73.8/83.8 | 73.8/83.3 | **1-stg R1 微优 +0.5** |

Tiny 和 Small **方向相反**: Tiny 2-stg R1 更好, Small 1-stg R1 更好。但差异 ≤ 0.5 R1, 方差范围内。mAP **完全持平**。

## 🔥 论文叙事升级

**GCN 零贡献 reconfirmed**:
- exp288 LGPA-only 1-stg 73.8/83.8 = exp285b Full Scaffold 73.8/83.8 (持平)
- exp289 LGPA-only 2-stg 73.8/83.3 ≈ Full Scaffold (mAP 持平, R1 -0.5)
- **可以去掉 GCN 简化模型 0 性能损失**

## srvC 状态

- exp289 FINAL @ 05:39:56 CST
- daemon 94420 自动触发 **exp269b** (Base Market PLBOA OFF full 120) @ ~05:40
- exp269b e2 running at 05:54, Loss 13.5, 正常 warmup
- 预期 exp269b FINAL ~11:40 CST
