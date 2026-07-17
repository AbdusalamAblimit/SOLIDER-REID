# exp274 monitor — Phase 3-A Small no-PSG baseline (Occ-Duke, seed 42)

- 机器: lab4090 (24G RTX 4090)
- 启动: 2026-04-20 ~19:30 CST
- Log: /home/afr/SOLIDER-REID/log/occluded_duke/exp274_psg0_s_od_s42/train_log.txt
- Config: configs/occluded_duke/prcv_best_small.yml + `MODEL.POSE_ENABLED=False`
- Scaffold: 纯 Swin-Small,pose 模块全关(走 build_transformer 路径绕开 pose_model 的 dead import)

## 对照(Phase 3-A 矩阵)

| Exp | Backbone | PSG stages | FINAL mAP/R1 |
|-----|---------|-----------|-------------|
| exp270 | Tiny | 无 | 59.2 / 68.4 |
| exp271 | Tiny | `[-1]` | 60.2 / 69.5 |
| exp272 | Tiny | `[-2,-1]` | **60.5 / 69.7** |
| exp273 | Tiny | `[-3,-2,-1]` | 进行中 e30 |
| **exp274 (本)** | **Small** | **无** | **e100: 68.3/77.3 🔄** |
| exp275 | Small | `[-1]` | queued |
| exp276 | Small | `[-2,-1]` | queued |
| exp277 | Small | `[-3,-2,-1]` | queued |

核心: exp274 是 Small 系列 baseline,和 exp270 形成 backbone 缩放对照。历史 Small baseline(no flip)~65.8,+default flip 应 ~67-68。**实际 e100 已 68.3/77.3 → 超预期上限。**

## 训练轨迹 (flip-test,eq_concat global)

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 45.7 | 55.5 | 70.2 | 75.7 |
| 20 | 50.8 | 59.1 | 71.6 | 77.3 |
| 30 | 59.8 | 68.5 | 81.3 | 84.9 |
| 40 | 62.8 | 72.2 | 84.0 | 87.7 |
| 50 | 65.0 | 73.9 | 85.8 | 88.8 |
| 60 | 65.6 | 74.5 | 86.4 | 89.5 |
| 70 | 66.9 | 75.6 | 87.1 | 90.5 |
| 80 | 67.7 | 76.5 | 87.4 | 90.1 |
| 90 | 68.0 | 77.3 | 87.6 | 90.5 |
| 100 | 68.3 | 77.3 | 87.9 | 91.0 | peak mAP |
| 110 | 68.1 | 76.8 | 87.8 | 90.9 | 轻微回撤 |
| **120 FINAL** | **68.1** | **76.8** | **87.8** | **90.9** | ckpt 21:34 CST |

- Speed: ~48s/epoch (Small + PSG 关),总训练时长 ~1h36min (19:58-21:34 CST wall clock)
- vs exp270 Tiny no-PSG 59.2/68.4: **Δ=+8.9/+8.4** (纯 backbone 容量收益)

## FINAL (21:34 CST)

- **mAP: 68.1%**, **Rank-1: 76.8%**, Rank-5: 87.8%, Rank-10: 90.9%
- 对照 exp270 Tiny no-PSG 59.2/68.4 → Δ=**+8.9/+8.4** (纯 Small vs Tiny backbone 容量差)
- vs e100 peak 68.3/77.3: 尾部 LR 低期轻微回撤 -0.2/-0.5 (正常训练噪声)
- Ckpt: `/home/afr/SOLIDER-REID/log/occluded_duke/exp274_psg0_s_od_s42/transformer_120.pth` (198MB)

## 结论

- Small no-PSG baseline **68.1/76.8** 超原预期上限 (预期 67-68)
- e100 已接近 peak (68.3),e120 FINAL 与 e110 持平 → 训练已饱和,尾部 epoch 带来边际收益约 0
- Phase 3-A Small 的 PSG 贡献应从此基线上算: exp275/276/277 目标是看能否超过 68.1/76.8
- auto-chain exp275 Small 1-stage PSG 等待触发 (daemon 3580255 监控 ckpt 出现)
