# exp273 monitor — Phase 3-A 3-stage PSG (Swin-Tiny @ Occ-Duke, seed 42)

- 机器: srvB
- 启动: pending (auto-chain from exp272 via queue_on_ckpt daemon)
- Log: /hy-tmp/log/occluded_duke/exp273_psg3_t_od_s42/train_log.txt
- Config: configs/occluded_duke/prcv_best_tiny.yml + CLI override
- Scaffold: **Swin-Tiny + PSG stages [-3,-2,-1]** (LGPA/GCN/OA-SD/PLBOA/ParAug 全部关)

## 对照(Phase 3-A 矩阵)

| Exp | PSG stages | FINAL mAP/R1 |
|-----|-----------|-------------|
| exp270 | 无 | 59.2 / 68.4 |
| exp271 | `[-1]` | 60.2 / 69.5 |
| exp272 | `[-2,-1]` | 进行中 |
| **exp273 (本)** | `[-3,-2,-1]` | pending |

核心: exp272 vs exp273 的 mAP 差就是增加 stage 1 PSG 的边际贡献。历史预期 3-stage ≤ 2-stage。

## 自动化状态

- 待挂 queue_on_ckpt daemon,等待 exp272 的 `transformer_120.pth` 出现
