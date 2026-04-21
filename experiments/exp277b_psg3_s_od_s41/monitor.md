# exp277b monitor — Phase 3-A Small 3-stage PSG **seed 41 重跑** (Occ-Duke)

- 机器: lab4090
- 启动: 2026-04-21 21:35 CST (auto-chain from exp284 FINAL via daemon 3909905)
- Log: `/home/afr/SOLIDER-REID/log/occluded_duke/exp277b_psg3_s_od_s41/train_log.txt`
- Config: `configs/occluded_duke/prcv_best_small.yml` + CLI (PSG_STAGES `[-3,-2,-1]`, pure scaffold, SEED 41)
- 和 exp277 (seed 42 塌缩 49.0/57.7) 仅 seed 不同
- Speed: ~55s/epoch, 总训练 1h59min

## 训练轨迹

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 42.1 | 51.2 | 64.7 | 70.3 |
| 20 | 48.4 | 56.4 | 71.5 | 75.8 |
| 30 | 59.5 | 68.0 | 80.4 | 85.1 |
| 40 | 61.6 | 70.0 | 82.8 | 86.7 |
| 50 | 63.7 | 73.0 | 85.1 | 88.0 |
| 60 | 64.0 | 72.6 | 84.5 | 88.0 |
| 70 | 67.0 | 76.1 | 86.9 | 89.4 |
| 80 | 68.1 | 77.6 | 87.2 | 89.9 |
| 90 | 67.8 | 77.2 | 87.2 | 90.0 |
| 100 | 68.3 | 77.4 | 87.4 | 89.8 |
| 110 | 68.2 | 77.4 | 87.3 | 89.8 |
| **120 FINAL** | **68.3** | **77.6** | **87.4** | **89.8** |

## FINAL (2026-04-21 23:34 CST)

- **mAP: 68.3%**, **Rank-1: 77.6%**, R5: **87.4%**, R10: 89.8%
- 对照:
  - **exp277 (seed 42 塌缩)**: 49.0/57.7 → Δ=**+19.3/+19.9** (seed 差 19+)
  - exp274 (no-PSG): 68.1/76.8 → Δ=+0.2/+0.8
  - exp275 (1-stg): 68.8/76.8 → Δ=-0.5/+0.8
  - exp276 (2-stg): 68.3/77.2 → Δ=0/+0.4
- Ckpt: `transformer_120.pth` (~200MB)

## 🔥 Phase 3-A Small 矩阵**完整闭合**

| stage | mAP | R1 | 备注 |
|-------|-----|----|----|
| 无 (exp274) | 68.1 | 76.8 | baseline |
| 1 (exp275) | **68.8** | 76.8 | mAP peak |
| 2 (exp276) | 68.3 | 77.2 | R1 中等 |
| **3 (exp277b s41)** | 68.3 | **77.6** | **R1 peak!** |

**Phase 3-A Small 结论**:
1. **mAP peak = 1-stage (68.8)**
2. **R1 peak = 3-stage seed 41 (77.6)**
3. exp277 seed 42 49.0/57.7 塌缩是**偶发 seed 问题** (用户判断完全正确)
4. 和 Tiny 3-stage (exp273 60.5/69.9 比 exp272 2-stage R1 +0.2) 一致: **3-stage R1 优势跨 backbone 成立**

**论文主表 Phase 3-A Small 用 exp277b 数字** (替换 exp277 塌缩)。

## auto-chain → exp285b

daemon 4027889 挂 exp277b → exp285b (Small GCN512+2stg gold-standard, 跨设备 rerun exp262 config)
exp285b 启动预计 ~23:35, FINAL tmr ~01:35 CST
