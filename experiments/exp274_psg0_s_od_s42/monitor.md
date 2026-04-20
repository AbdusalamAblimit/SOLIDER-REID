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
| exp272 | Tiny | `[-2,-1]` | 进行中 e90 |
| exp273 | Tiny | `[-3,-2,-1]` | pending |
| **exp274 (本)** | **Small** | **无** | **pending** |
| exp275 | Small | `[-1]` | pending |
| exp276 | Small | `[-2,-1]` | pending |
| exp277 | Small | `[-3,-2,-1]` | pending |

核心: exp274 是 Small 系列 baseline,和 exp270 形成 backbone 缩放对照。历史 Small baseline(no flip)~65.8,+default flip 应 ~67-68。

## 预期

- mAP: ~67-68
- R1: ~76-78
- 时长: ~3h (4090 + Small pure)
