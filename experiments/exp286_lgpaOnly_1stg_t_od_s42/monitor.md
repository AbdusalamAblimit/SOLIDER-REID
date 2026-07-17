# exp286 monitor — Phase 3-C Tiny LGPA-only + 1-stage PSG (Occ-Duke, seed 42)

- 机器: srvC (i-2.gpushare.com:25551, 5060 Ti 16G)
- 启动: 2026-04-20 23:32 CST (手动启动, exp266 silent exit 后利用空闲)
- Log: `/hy-tmp/log/occluded_duke/exp286_lgpaOnly_1stg_t_od_s42/train_log.txt`
- Config: `configs/occluded_duke/prcv_best_tiny.yml` + CLI override
  - `MODEL.POSE_SKELETON_GCN False` (关闭结构分支)
  - `MODEL.POSE_PSG_STAGES [-1]` (1-stage PSG)
- Scaffold: Swin-Tiny + LGPA + OA-SD + ParAug + LOWER_BODY_OCC **- GCN** + 1-stage PSG
- Speed: ~305s/epoch, 总训练 10h31min (23:32 → 10:03)

## 对照 (Phase 3-C Tiny 2 格 + Phase 3-B 对比)

| Exp | scaffold | PSG | GCN | mAP / R1 |
|-----|----------|-----|-----|----------|
| **exp286 (本)** | Tiny Full - GCN | `[-1]` | False | **66.0 / 76.6** |
| exp287 | Tiny Full - GCN | `[-2,-1]` | False | 进行中 (next) |
| exp278 | Tiny Full | `[-1]` | 256 | e110: 65.7/76.9 (进行中) |
| exp261 (=exp281) | Tiny Full | `[-2,-1]` | 512 | 65.9 / 77.4 |

## 训练轨迹 (flip-test, eq_concat global)

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 43.0 | 56.8 | 72.3 | 77.1 |
| 20 | 52.1 | 66.2 | 79.7 | 84.3 |
| 30 | 58.1 | 70.1 | 82.0 | 85.4 |
| 40 | 61.4 | 72.7 | 84.4 | 87.6 |
| 50 | 62.8 | 74.6 | 84.9 | 88.0 |
| 60 | 64.3 | 76.2 | 86.5 | 89.2 |
| 70 | 64.6 | 75.7 | 86.4 | 89.1 |
| 80 | 65.4 | 76.6 | 86.4 | 89.5 |
| 90 | 65.7 | 76.4 | 87.1 | 89.4 |
| 100 | 66.0 | 77.1 | 86.6 | 89.5 |
| 110 | 65.9 | 76.6 | 86.3 | 89.6 |
| **120 FINAL** | **66.0** | **76.6** | **86.4** | **89.7** |

## FINAL (2026-04-21 10:03 CST)

- **mAP: 66.0%**, **Rank-1: 76.6%**, R5: 86.4%, R10: 89.7%
- 对照:
  - **exp261 Tiny Full Scaffold GCN512+2stg**: 65.9/77.4 → Δ=**+0.1 mAP / -0.8 R1**
  - **exp271 Tiny pure PSG 1-stage**: 60.2/69.5 → Δ=+5.8/+7.1 (scaffold 整体收益)
- Ckpt: `transformer_120.pth`

## 🔥 核心论文结论 (Phase 3-C 首个 Tiny FINAL)

**GCN 对 Tiny backbone 贡献几乎为 0 (mAP) 或微负 (R1)**:
- **LGPA-only + 1-stage PSG (exp286): 66.0/76.6**
- **Full Scaffold GCN512+2stg (exp261): 65.9/77.4**
- Δ = +0.1 / -0.8

**论文解读**:
1. Tiny backbone 的 semantic branch (LGPA + OA-SD + ParAug + LOWER_BODY_OCC) 已提供全部关键增益
2. **GCN + 2-stage PSG 相比 LGPA-only + 1-stage PSG 只带来 +0.8 R1 微增**, mAP 无增益
3. 结合 exp278 Tiny Full GCN256+1stg pending (e110 65.7/76.9) → Tiny 下 GCN cap / PSG stage 对 mAP 都无显著贡献
4. **论文 default 可考虑 LGPA-only (Tiny)**, 节省 GCN 参数 + 训练复杂度
5. 等 exp287 FINAL (~tmr 00:30) 验证 "2-stage PSG 在 LGPA-only 下有无额外贡献"

## Phase 3-C chain

- daemon 59846 挂 exp286 → exp287 (Tiny LGPA-only 2-stage)
- exp287 已 auto-launched (Monitor 已确认)
- Phase 3-C Small exp288/289 尚未规划, 待 Phase 3-B Small 完成后考虑
