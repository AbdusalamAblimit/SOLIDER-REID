# exp282 monitor — Phase 3-B Small Full Scaffold GCN256+1stg (Occ-Duke, seed 42)

- 机器: lab4090
- 启动: 2026-04-21 03:48 CST (auto-chain from exp277 FINAL via v2 daemon 3674925)
- Log: `/home/afr/SOLIDER-REID/log/occluded_duke/exp282_gcn256_1stg_s_od_s42/train_log.txt`
- Config: `configs/occluded_duke/prcv_best_small.yml` + CLI override
  - `MODEL.POSE_GCN_HIDDEN 256` (default 512)
  - `MODEL.POSE_PSG_STAGES [-1]` (default `[-2,-1]`)
- Scaffold: Swin-Small + Full Scaffold (LGPA + GCN256 + OA-SD + ParAug + LOWER_BODY_OCC) + 1-stage PSG
- Speed: ~165s/epoch, 总训练 5h45min (03:48 → 09:33)

## 对照 (Phase 3-B Small 2×2 矩阵)

| Exp | GCN_HIDDEN | PSG_STAGES | mAP / R1 | 设备 |
|-----|-----------|------------|----------|------|
| **exp282 (本)** | **256** | **`[-1]`** | **73.7 / 83.9** | lab4090 |
| exp283 | 256 | `[-2,-1]` | 进行中 | lab4090 (next) |
| exp284 | 512 | `[-1]` | queued | lab4090 |
| exp285 ≡ exp262 | 512 | `[-2,-1]` | 73.8 / 83.1 | srvA (old code) |
| exp285b | 512 | `[-2,-1]` | queued | lab4090 (same-device gold standard) |

## 训练轨迹 (flip-test, eq_concat global)

| Epoch | mAP | R1 | R5 | R10 | 备注 vs exp262 |
|-------|-----|----|----|----|------------------|
| 10 | 56.0 | 68.1 | 82.3 | 86.2 | 起步强 (+11.5 vs pure exp275 e10 44.5) |
| 20 | 61.7 | 72.6 | 84.6 | 87.3 | |
| 30 | 68.9 | 80.0 | 88.6 | 90.6 | |
| 40 | 69.3 | 79.7 | 88.7 | 90.8 | |
| 50 | 71.5 | 81.8 | 89.7 | 92.0 | R1 持平 exp262 FINAL 83.1 |
| 60 | 72.5 | 82.9 | 90.2 | 92.1 | |
| 70 | 72.7 | 83.1 | 90.5 | 92.2 | **R1 持平 exp262!** |
| 80 | **73.5** | 83.2 | 90.5 | 92.4 | R1 超 exp262 +0.1 |
| 90 | 73.5 | **83.7** | 90.5 | 92.5 | **R1 超 exp262 +0.6** |
| 100 | 73.6 | 83.9 | 90.4 | 92.5 | R1 超 +0.8 |
| 110 | 73.7 | 83.9 | 90.5 | 92.6 | |
| **120 FINAL** | **73.7** | **83.9** | **90.5** | **92.5** | |

## FINAL (2026-04-21 09:33 CST)

- **mAP: 73.7%**, **Rank-1: 83.9%**, R5: 90.5%, R10: 92.5%
- 对照 exp262 (= exp285 slot, Phase 1 Small Full GCN512+2stg): **73.8 / 83.1 / 90.2 / 92.2**
- **Δ = -0.1 / +0.8 / +0.3 / +0.3** (mAP 几乎持平, R1 明显超越)
- Ckpt: `transformer_120.pth` (226MB, 比 exp275 pure PSG ckpt 大很多, 因 LGPA+OA-SD+GCN+OAB 权重)

## 🔥 核心论文结论 (Phase 3-B 首个 Full Small FINAL)

**Small Full Scaffold 配置下, low-cap (GCN256 + 1-stage PSG) ≥ high-cap (GCN512 + 2-stage PSG):**
- mAP: 73.7 (GCN256+1stg) ≈ 73.8 (GCN512+2stg)  
- R1: **83.9 > 83.1** (low-cap 反超 0.8)

### 论文解读 (Phase 3-B 初步结论)

1. **Small + Full Scaffold 的 PSG/GCN 容量已饱和**: 更多 stage 或更大 GCN 边际贡献小于方差
2. **GCN256 + 1-stage PSG 可作 default scaffold** (参数少 + R1 等效)
3. exp262 srvA 73.8/83.1 作为跨设备历史数字保留, exp285b 将给出同设备 gold-standard 验证

## auto-chain → exp283

daemon 3674926 已挂, exp282/transformer_120.pth 已生成 @ 01:31 UTC (09:31 CST), daemon 在 poll 等 python process exit + 20s safety 后 launch exp283 (Small GCN256+2stg)。

预计 exp283 启动 ~09:34-36 CST, FINAL ~15:30 CST。
