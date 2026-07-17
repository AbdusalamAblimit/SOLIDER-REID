# exp279 monitor — Phase 3-B Tiny Full Scaffold GCN256+2stg (Occ-Duke, seed 42)

- 机器: srvB
- 启动: 2026-04-21 10:44 CST (auto-chain from exp278 FINAL via daemon 70448)
- Log: `/hy-tmp/log/occluded_duke/exp279_gcn256_2stg_t_od_s42/train_log.txt`
- Config: `configs/occluded_duke/prcv_best_tiny.yml` + CLI `MODEL.POSE_GCN_HIDDEN 256`
- Scaffold: Swin-Tiny + Full Scaffold (LGPA + GCN256 + OA-SD + ParAug + LOWER_BODY_OCC) + 2-stage PSG `[-2,-1]`
- Speed: ~310s/epoch, 总训练 10h48min (10:44 → 21:32)

## 训练轨迹

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 43.0 | 56.6 | 72.0 | 77.6 |
| 20 | 53.7 | 66.3 | 80.5 | 84.4 |
| 30 | 59.5 | 71.6 | 84.0 | 87.6 |
| 40 | 61.6 | 74.0 | 85.0 | 87.9 |
| 50 | 63.5 | 74.5 | 86.1 | 88.9 |
| 60 | 63.9 | 76.0 | 86.3 | 88.8 |
| 70 | 64.7 | 76.4 | 86.7 | 89.1 |
| 80 | 65.0 | 76.4 | 86.4 | 89.6 |
| 90 | 65.4 | 77.0 | 86.6 | 89.6 |
| 100 | 65.7 | 77.4 | 86.8 | 89.9 |
| 110 | 65.6 | 76.8 | 86.4 | 90.1 |
| **120 FINAL** | **65.7** | **76.9** | **86.6** | **90.1** |

## FINAL (2026-04-21 21:32 CST)

- **mAP: 65.7%**, **Rank-1: 76.9%**, R5: 86.6%, R10: 90.1%
- 对照:
  - exp278 GCN256+1stg: 65.7/76.7 → Δ=**0 / +0.2** (mAP 持平, R1 微优)
  - exp261 (=exp281) GCN512+2stg: 65.9/77.4 → Δ=-0.2/-0.5
  - exp286 LGPA-only 1stg: 66.0/76.6 → Δ=-0.3/+0.3
- Ckpt: `transformer_120.pth` (141MB)

## 🔥 Phase 3-B Tiny 3/4 — 核心结论

| | GCN256 | GCN512 |
|---|---|---|
| PSG `[-1]` | **exp278 65.7/76.7** | exp280 pending |
| PSG `[-2,-1]` | **exp279 65.7/76.9** | exp261 65.9/77.4 |

**Tiny 观察**:
1. **GCN256 下 1-stg ≈ 2-stg** (mAP 65.7=65.7, R1 +0.2 微差)
2. **GCN512 下 1-stg vs 2-stg**: 待 exp280 FINAL 对照
3. **2-stg PSG 在 Tiny 上的 R1 增益极小** (+0.2 on GCN256, TBD on GCN512)
4. 和 Small Phase 3-B (exp282 GCN256+1stg 83.9 R1 最强) 模式类似: **low-cap + low-stage 已经足够**

## auto-chain → exp280

daemon 83434 (replaces original 70449) 挂 exp279 → exp280 with TEST.IMS_PER_BATCH 128
exp280 启动 ~21:33-35, FINAL ~tmr 07:35 CST
