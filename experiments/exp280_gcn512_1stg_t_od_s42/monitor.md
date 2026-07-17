# exp280 monitor — Phase 3-B Tiny Full Scaffold GCN512+1stg (Occ-Duke, seed 42)

- 机器: srvB
- 启动: 2026-04-21 21:45 CST (auto-chain from exp279 FINAL via daemon 83434 with TEST.IMS_PER_BATCH 128)
- Log: `/hy-tmp/log/occluded_duke/exp280_gcn512_1stg_t_od_s42/train_log.txt`
- Config: `configs/occluded_duke/prcv_best_tiny.yml` + CLI `MODEL.POSE_PSG_STAGES [-1] TEST.IMS_PER_BATCH 128`
- Scaffold: Swin-Tiny + Full Scaffold (LGPA + **GCN512** + OA-SD + ParAug + LOWER_BODY_OCC) + 1-stage PSG
- Speed: ~305s/epoch, 总训练 10h22min

## 训练轨迹 (flip-test, eq_concat global)

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 43.7 | 56.1 | 72.8 | 78.0 |
| 20 | 51.5 | 65.1 | 79.9 | 83.8 |
| 30 | 59.1 | 71.3 | 83.2 | 87.4 |
| 40 | 60.8 | 72.5 | 84.1 | 87.6 |
| 50 | 62.8 | 75.2 | 86.0 | 88.7 |
| 60 | 63.8 | 76.2 | 86.4 | 89.2 |
| 70 | 64.9 | 76.5 | 86.9 | 89.6 |
| 80 | 65.3 | 76.6 | 87.1 | 89.4 |
| 90 | 65.4 | 76.2 | 86.7 | 89.9 |
| 100 | 65.5 | 76.9 | 87.0 | 89.5 |
| 110 | 65.6 | 76.2 | 86.8 | 89.6 |
| **120 FINAL** | **65.7** | **76.2** | **86.7** | **89.7** |

## FINAL (2026-04-22 08:07:50 CST)

- **mAP: 65.7%**, **Rank-1: 76.2%**, R5: 86.7%, R10: 89.7%
- 对照:
  - **exp261 (= exp281) GCN512+2stg**: 65.9/77.4 → Δ=**-0.2/-1.2** (2-stg R1 明显更优)
  - exp278 GCN256+1stg: 65.7/76.7 → Δ=**0/-0.5** (GCN cap 持平, R1 反弱)
  - exp279 GCN256+2stg: 65.7/76.9 → Δ=**0/-0.7**
  - exp286 LGPA-only 1stg: 66.0/76.6 → Δ=-0.3/-0.4
- Ckpt: `transformer_120.pth` (142MB)

## 🔥 Phase 3-B Tiny 2×2 完整闭合

| | GCN256 | GCN512 |
|---|---|---|
| PSG `[-1]` | 65.7/76.7 (exp278) | **65.7/76.2** (exp280, weakest R1) |
| PSG `[-2,-1]` | 65.7/76.9 (exp279) | **65.9/77.4** (exp261, best) |

**Tiny 核心结论**:
1. **GCN512+1stg (exp280) 是 2×2 最弱 R1 格** (76.2), 和 Small 2×2 GCN512+1stg (exp284 73.4/82.9) 同模式
2. **Tiny GCN512 必须配 2-stg** 才 R1 最强 (exp261 77.4)
3. **GCN256 下 1-stg ≈ 2-stg** (65.7=65.7, R1 差 0.2), 低 cap 对 stage 不敏感
4. **方差 ≤ 0.2 mAP / 1.2 R1** — 所有 4 格非常接近, 和 Small 2×2 (方差 ≤ 0.4/1.0) 相当

## 🎯 Tiny vs Small 同模式

| Backbone | 最弱 2×2 格 | 数字 | 最强 2×2 格 | 数字 |
|----------|-------------|------|-------------|------|
| Tiny | GCN512+1stg (exp280) | 65.7/**76.2** | GCN512+2stg (exp261) | **65.9**/77.4 |
| Small | GCN512+1stg (exp284) | 73.4/82.9 | GCN512+2stg (exp285b) | **73.8/83.8** |

**跨 backbone 一致**: 大 GCN 需 2-stg, 1-stg 浪费 GCN 容量。

## srvB idle (2026-04-22 08:05 CST)

- Phase 3-B Tiny 4/4 FINAL, 无 chain daemon → srvB GPU 空闲
- 可选: Task #6 LR4 vs LR8 / Task #12 MaxSim eval (等 srvC Phase 3-C 完成再起)
