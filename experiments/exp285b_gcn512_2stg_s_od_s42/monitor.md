# exp285b monitor — Phase 3-B Small exp262 同设备 rerun (Occ-Duke, seed 42)

- 机器: lab4090 (4090 24G, mmpose-abu env)
- 启动: 2026-04-21 23:35 CST (auto-chain from exp277b FINAL via daemon 4027889)
- Log: `/home/afr/SOLIDER-REID/log/occluded_duke/exp285b_gcn512_2stg_s_od_s42/train_log.txt`
- Config: `configs/occluded_duke/prcv_best_small.yml` default (GCN512 + 2-stg PSG + Full Scaffold + seed 42)
- Scaffold: 和 exp262 完全相同
- Speed: ~176s/epoch, 总训练 6h30min

## 训练轨迹 (flip-test, eq_concat global)

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 56.1 | 68.4 | 82.0 | 85.9 |
| 20 | 62.6 | 74.2 | 85.0 | 87.4 |
| 30 | 69.4 | 81.0 | 88.8 | 91.0 |
| 40 | 69.5 | 80.9 | 89.1 | 91.0 |
| 50 | 71.3 | 80.8 | 89.8 | 91.9 |
| 60 | 72.5 | 83.0 | 90.4 | 92.4 |
| 70 | 72.7 | 83.0 | 90.4 | 92.4 |
| 80 | 73.4 | 83.2 | 90.9 | 92.6 |
| 90 | 73.4 | 83.6 | 90.4 | 92.8 |
| 100 | 73.6 | 83.6 | 90.5 | 92.7 |
| 110 | 73.8 | 83.7 | 90.7 | 92.8 |
| **120 FINAL** | **73.8** | **83.8** | **90.7** | **92.7** |

## FINAL (2026-04-22 06:04 CST)

- **mAP: 73.8%**, **Rank-1: 83.8%**, R5: 90.7%, R10: 92.7%
- 对照:
  - **exp262 (srvA old code seed 42) FINAL**: 73.8/83.1 → Δ=**0 / +0.7** (mAP 持平!, R1 +0.7)
  - exp282 (lab4090 GCN256+1stg): 73.7/83.9 → Δ=+0.1/-0.1 (几乎持平)
  - exp283 (lab4090 GCN256+2stg): 73.5/83.2 → Δ=+0.3/+0.6
  - exp284 (lab4090 GCN512+1stg): 73.4/82.9 → Δ=+0.4/+0.9

## 🔥 Phase 3-B Small 矩阵最终闭合 (全 lab4090 同设备)

| | GCN256 | GCN512 |
|---|---|---|
| PSG `[-1]` | **73.7 / 83.9** (exp282) | 73.4 / 82.9 (exp284) |
| PSG `[-2,-1]` | 73.5 / 83.2 (exp283) | **73.8 / 83.8** (exp285b) |

## 🎯 论文结论

1. **exp262 数字跨设备验证**: mAP 完全持平 (73.8), R1 +0.7 (83.1 → 83.8) = **lab4090 同设备确认 exp262 可信, 选用 exp285b 数字更严谨**
2. **GCN512+2stg 和 GCN256+1stg 是双 sweet spot** (mAP 73.7-73.8, R1 83.8-83.9 差 0.1)
3. **GCN512+1stg (exp284) 反而最弱** (73.4/82.9) — 大 GCN 需 2-stg 配套
4. **GCN256+2stg (exp283)** 中等 (73.5/83.2)

**Phase 3-B Small 论文主表用 exp285b 73.8/83.8** (gold-standard,替换 exp262 srvA)。

## lab4090 空闲

- Phase 3-B Small chain 全部 FINAL
- Phase 3-A Small 含 exp277b s41 也完成
- lab4090 GPU 空闲 → 批量 MaxSim+flip re-eval (Task #12) 可在 lab4090 跑 Small ckpts
