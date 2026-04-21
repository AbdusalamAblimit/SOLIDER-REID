# exp284 monitor — Phase 3-B Small Full Scaffold GCN512+1stg (Occ-Duke, seed 42)

- 机器: lab4090
- 启动: 2026-04-21 15:40 CST (auto-chain from exp283 FINAL via daemon 3674927)
- Log: `/home/afr/SOLIDER-REID/log/occluded_duke/exp284_gcn512_1stg_s_od_s42/train_log.txt`
- Config: `configs/occluded_duke/prcv_best_small.yml` + CLI `MODEL.POSE_PSG_STAGES [-1]`
- Scaffold: Swin-Small + Full Scaffold (LGPA + **GCN512** + OA-SD + ParAug + LOWER_BODY_OCC) + 1-stage PSG
- Speed: ~164s/epoch, 总训练 5h43min (15:40 → 21:23)

## 对照 (Phase 3-B Small 2×2 矩阵 — 4/4 完整)

| Exp | GCN_HIDDEN | PSG_STAGES | mAP / R1 |
|-----|-----------|------------|----------|
| exp282 | 256 | `[-1]` | **73.7 / 83.9** |
| exp283 | 256 | `[-2,-1]` | 73.5 / 83.2 |
| **exp284 (本)** | **512** | **`[-1]`** | **73.4 / 82.9** |
| exp285 ≡ exp262 | 512 | `[-2,-1]` | 73.8 / 83.1 (srvA 历史, exp285b 替换 pending) |

## 训练轨迹

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 53.9 | 63.8 | 80.2 | 84.7 |
| 20 | 61.9 | 73.0 | 84.8 | 87.8 |
| 30 | 68.2 | 78.8 | 87.8 | 90.6 |
| 40 | 68.6 | 79.5 | 87.8 | 90.1 |
| 50 | 71.7 | 82.2 | 90.0 | 91.6 |
| 60 | 71.4 | 81.5 | 89.6 | 91.4 |
| 70 | 72.2 | 81.9 | 89.6 | 91.8 |
| 80 | 73.1 | 83.3 | 90.2 | 92.3 |
| 90 | 72.9 | 82.6 | 89.8 | 92.1 |
| 100 | 73.3 | 83.1 | 89.8 | 92.0 |
| 110 | 73.4 | 82.9 | 90.0 | 92.2 |
| **120 FINAL** | **73.4** | **82.9** | **89.9** | **92.2** |

## FINAL (2026-04-21 21:23 CST)

- **mAP: 73.4%**, **Rank-1: 82.9%**, R5: 89.9%, R10: 92.2%
- 对照:
  - exp262 (Full GCN512+2stg) 73.8/83.1: Δ=**-0.4/-0.2**
  - exp282 (Full GCN256+1stg) 73.7/83.9: Δ=**-0.3/-1.0**
  - exp283 (Full GCN256+2stg) 73.5/83.2: Δ=-0.1/-0.3
- Ckpt: `transformer_120.pth` (228MB)

## 🔥 Phase 3-B Small 2×2 完整结论

| | GCN256 | GCN512 |
|---|---|---|
| PSG `[-1]` | **73.7/83.9** (best R1) | 73.4/82.9 |
| PSG `[-2,-1]` | 73.5/83.2 | **73.8/83.1** (best mAP, =exp262) |

**核心发现**:
1. **方差 ≤ 0.4 mAP / 1.0 R1** — 所有 4 格非常接近
2. **R1 最强: GCN256+1stg (exp282 83.9)** — 低容量 + 少 stage 反而最强
3. **mAP 最强: GCN512+2stg (exp262 73.8)** — 高容量 + 多 stage 更稳
4. **GCN512+1stg (本 exp284) 反而最弱** — 可能大 GCN 需要 2-stg 才完整 exploit
5. **论文结论**: GCN256+1stg 和 GCN512+2stg 是 "light vs heavy" 两个最优配置点, 中间 (GCN512+1stg) 反而不佳

## auto-chain → exp277b

daemon 3909905 将触发 exp277b (Small pure PSG 3-stg seed 41, 用户指示 exp277 塌缩重跑)
