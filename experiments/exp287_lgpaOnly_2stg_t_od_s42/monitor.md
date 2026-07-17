# exp287 monitor — Phase 3-C Tiny LGPA-only + 2-stage PSG (Occ-Duke, seed 42)

- 机器: srvC (5060 Ti 16G)
- 启动: 2026-04-21 10:04 CST (auto-chain from exp286 FINAL via daemon 59846)
- Log: `/hy-tmp/log/occluded_duke/exp287_lgpaOnly_2stg_t_od_s42/train_log.txt`
- Config: `configs/occluded_duke/prcv_best_tiny.yml` + CLI `MODEL.POSE_SKELETON_GCN False`
- Scaffold: Swin-Tiny + LGPA + OA-SD + ParAug + LOWER_BODY_OCC - GCN + PSG `[-2,-1]`
- Speed: ~305s/epoch, 总训练 10h44min (10:04 → 20:48)

## 对照 (Phase 3-C Tiny 2/2 完整 + Phase 3-B 相关)

| Exp | scaffold | PSG | GCN | mAP / R1 |
|-----|---------|-----|-----|----------|
| exp286 | Tiny Full-GCN | `[-1]` | False | 66.0 / 76.6 |
| **exp287 (本)** | Tiny Full-GCN | `[-2,-1]` | False | **65.9 / 77.0** |
| exp278 | Tiny Full | `[-1]` | 256 | 65.7 / 76.7 |
| exp279 | Tiny Full | `[-2,-1]` | 256 | 进行中 e110 65.6/76.8 |
| exp261 | Tiny Full | `[-2,-1]` | 512 | 65.9 / 77.4 |

## 训练轨迹

| Epoch | mAP | R1 |
|-------|-----|----|
| 10 | 42.1 | 56.6 |
| 20 | 53.2 | 65.9 |
| 30 | 59.3 | 71.1 |
| 40 | 61.0 | 72.7 |
| 50 | 63.5 | 74.7 |
| 60 | 63.9 | 75.2 |
| 70 | 65.1 | 76.9 |
| 80 | 65.3 | 76.8 |
| 90 | 65.2 | 76.4 |
| 100 | 65.7 | 77.2 |
| 110 | 65.7 | 76.8 |
| **120 FINAL** | **65.9** | **77.0** |

## FINAL (2026-04-21 20:48 CST)

- **mAP: 65.9%**, **Rank-1: 77.0%**, R5: 87.0%, R10: 89.7%
- 对照:
  - **exp286 (LGPA-only 1stg)**: 66.0/76.6 → Δ=**-0.1 / +0.4** (2-stg R1 微优)
  - **exp261 (Full Scaffold GCN512+2stg)**: 65.9/77.4 → Δ=**0 / -0.4** (无 GCN 持平 mAP R1 微低)
- Ckpt: `transformer_120.pth`

## 🔥 Phase 3-C Tiny 2/2 完整结论

| scaffold | PSG | mAP | R1 |
|----------|-----|-----|----|
| LGPA-only (Full-GCN) | `[-1]` | 66.0 | 76.6 |
| LGPA-only (Full-GCN) | `[-2,-1]` | 65.9 | **77.0** |
| Full Scaffold (含 GCN512) | `[-2,-1]` | 65.9 | 77.4 |

**论文结论**:
1. **GCN 对 Tiny 主要贡献 R1 (+0.4)**, mAP 无贡献 (65.9 = 65.9 = 66.0 差 0.1)
2. **LGPA-only 已达 mAP 上限**, GCN 是"R1 锦上添花"
3. **2-stg PSG 在 LGPA-only 下 R1 微优 1-stg** (+0.4, 和 Phase 3-B 趋势一致)

## srvC 后续

- Phase 3-C Tiny chain 结束, srvC 空闲
- 立即接 **exp288 Small LGPA-only 1-stg** (Phase 3-C Small)
