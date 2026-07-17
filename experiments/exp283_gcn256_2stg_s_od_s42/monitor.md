# exp283 monitor — Phase 3-B Small Full Scaffold GCN256+2stg (Occ-Duke, seed 42)

- 机器: lab4090
- 启动: 2026-04-21 09:37 CST (auto-chain from exp282 FINAL via daemon 3674926)
- Log: `/home/afr/SOLIDER-REID/log/occluded_duke/exp283_gcn256_2stg_s_od_s42/train_log.txt`
- Config: `configs/occluded_duke/prcv_best_small.yml` + CLI `MODEL.POSE_GCN_HIDDEN 256`
- Scaffold: Swin-Small + Full Scaffold (LGPA + GCN256 + OA-SD + ParAug + LOWER_BODY_OCC) + 2-stage PSG `[-2,-1]`
- Speed: ~175s/epoch, 总训练 6h (09:37 → 15:39)

## 对照 (Phase 3-B Small 2×2 矩阵)

| Exp | GCN_HIDDEN | PSG_STAGES | mAP / R1 |
|-----|-----------|------------|----------|
| exp282 | 256 | `[-1]` | **73.7 / 83.9** ✅ |
| **exp283 (本)** | **256** | **`[-2,-1]`** | **73.5 / 83.2** |
| exp284 | 512 | `[-1]` | pending (next) |
| exp285 ≡ exp262 | 512 | `[-2,-1]` | 73.8 / 83.1 (srvA 历史, exp285b 替换) |

## 训练轨迹 (flip-test, eq_concat global)

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 56.5 | 67.6 | 81.0 | 85.8 |
| 20 | 62.2 | 73.4 | 84.7 | 87.8 |
| 30 | 69.2 | 81.1 | 89.3 | 91.6 |
| 40 | 68.8 | 80.4 | 89.0 | 91.4 |
| 50 | 71.5 | 81.6 | 89.3 | 91.9 |
| 60 | 72.3 | 82.5 | 90.3 | 92.3 |
| 70 | 72.3 | 82.9 | 90.1 | 92.1 |
| 80 | 73.1 | 83.2 | 90.2 | 92.0 |
| 90 | 73.2 | 83.4 | 90.8 | 92.6 |
| 100 | 73.3 | 83.0 | 90.5 | 92.4 |
| 110 | 73.5 | 83.2 | 90.6 | 92.4 |
| **120 FINAL** | **73.5** | **83.2** | **90.7** | **92.5** |

## FINAL (2026-04-21 15:38 CST)

- **mAP: 73.5%**, **Rank-1: 83.2%**, R5: 90.7%, R10: 92.5%
- 对照:
  - exp262 (=exp285) Full GCN512+2stg: 73.8/83.1 → Δ=**-0.3/+0.1** (R1 持平)
  - exp282 Full GCN256+1stg: 73.7/83.9 → Δ=**-0.2/-0.7** (低容量 1-stg 更强)
- Ckpt: `transformer_120.pth` (227MB)

## 🔥 Phase 3-B Small 3/4 FINAL — 核心结论

**Small Full Scaffold 的 PSG/GCN 容量影响很小** (<0.3 mAP 方差):

| | GCN256 | GCN512 |
|---|---|---|
| PSG `[-1]` | **exp282: 73.7 / 83.9** | exp284 pending |
| PSG `[-2,-1]` | **exp283: 73.5 / 83.2** | exp262: 73.8 / 83.1 |

观察:
1. **GCN256 下 1-stg ≥ 2-stg** (73.7 > 73.5 in mAP, R1 差 0.7)
2. **2-stg 条件下 GCN512 > GCN256** (73.8 vs 73.5 in mAP, 0.3 差)
3. **exp282 (GCN256+1stg) R1 最高 83.9** — 最轻量 + R1 最强
4. 和 Tiny Phase 3-B 结论一致: **low-cap 足够**

## auto-chain → exp284

daemon 3674927 挂 exp283 → exp284 (Small Full GCN512+1stg)
exp284 预计启动 ~15:40, FINAL ~21:40 CST
