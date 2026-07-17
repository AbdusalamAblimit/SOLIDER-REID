# exp278 monitor — Phase 3-B Tiny Full Scaffold GCN256+1stg (Occ-Duke, seed 42)

- 机器: srvB
- 启动: 2026-04-21 ~00:12 CST (auto-chain from exp273 FINAL via daemon 70447)
- Log: `/hy-tmp/log/occluded_duke/exp278_gcn256_1stg_t_od_s42/train_log.txt`
- Config: `configs/occluded_duke/prcv_best_tiny.yml` + CLI override
  - `MODEL.POSE_GCN_HIDDEN 256` (default 512)
  - `MODEL.POSE_PSG_STAGES [-1]` (default `[-2,-1]`)
- Scaffold: Swin-Tiny + Full Scaffold (LGPA + GCN256 + OA-SD + ParAug + LOWER_BODY_OCC) + 1-stage PSG
- Speed: ~305s/epoch, 总训练 10h30min

## 对照 (Phase 3-B Tiny 2×2 矩阵)

| Exp | GCN_HIDDEN | PSG_STAGES | mAP / R1 |
|-----|-----------|------------|----------|
| **exp278 (本)** | **256** | **`[-1]`** | **65.7 / 76.7** |
| exp279 | 256 | `[-2,-1]` | 进行中 (next) |
| exp280 | 512 | `[-1]` | queued |
| exp281 ≡ exp261 | 512 | `[-2,-1]` | 65.9 / 77.4 |

## 训练轨迹 (flip-test, eq_concat global)

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 44.2 | 58.5 | 73.3 | 78.6 |
| 20 | 52.5 | 65.7 | 79.8 | 84.2 |
| 30 | 59.8 | 70.9 | 84.0 | 87.6 |
| 40 | 61.6 | 73.5 | 85.0 | 87.9 |
| 50 | 63.0 | 74.1 | 86.1 | 88.9 |
| 60 | 63.5 | 75.2 | 86.3 | 88.8 |
| 70 | 65.2 | 76.8 | 86.7 | 89.1 |
| 80 | 65.0 | 76.7 | 86.6 | 89.2 |
| 90 | 65.2 | 76.5 | 86.5 | 89.5 |
| 100 | 65.6 | 76.5 | 86.7 | 89.7 |
| 110 | 65.7 | 76.9 | 86.7 | 89.7 |
| **120 FINAL** | **65.7** | **76.7** | **86.7** | **89.6** |

## FINAL (2026-04-21 10:42 CST)

- **mAP: 65.7%**, **Rank-1: 76.7%**, R5: 86.7%, R10: 89.6%
- 对照:
  - **exp261 Tiny Full GCN512+2stg**: 65.9/77.4 → Δ=**-0.2/-0.7**
  - **exp286 Tiny LGPA-only 1stg (no GCN)**: 66.0/76.6 → Δ=**-0.3/+0.1**

## 🔥 核心结论 (Tiny 上 GCN 作用存疑)

**Tiny backbone 下 GCN256+1stg 不如 LGPA-only (no GCN):**
- exp278 (GCN256): 65.7/76.7
- exp286 (no GCN): **66.0/76.6** → mAP 微超 exp278 +0.3, R1 -0.1
- exp261 (GCN512): 65.9/77.4 → R1 最高 (+0.7 vs exp278)

**论文解读 (Tiny)**:
1. **GCN 容量必须至少 512 才带来 R1 增益** (256 低容量反而无效)
2. Tiny 的 mAP 已 saturate 在 65.7-66.0, 无论 scaffold 细节
3. Tiny default 可用 **LGPA-only + 1-stage PSG** (mAP 最高 + 无 GCN 参数开销)
4. 等 exp279 (Tiny GCN256+2stg) + exp280 (Tiny GCN512+1stg) FINAL 完成 2×2 闭合验证

## auto-chain → exp279

daemon 70448 挂 exp278 → exp279 (Tiny Full GCN256+2stg)
exp279 预计 ~10:42 + daemon poll delay → 启动 ~10:45, FINAL tmr ~21:15 CST (5-6h + 9-10h = 15h? 实际 305s × 120 = 10h15min)
