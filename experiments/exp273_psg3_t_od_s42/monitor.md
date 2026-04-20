# exp273 monitor — Phase 3-A 3-stage PSG (Swin-Tiny @ Occ-Duke, seed 42)

- 机器: srvB
- 启动: 2026-04-20 20:19 (auto-chain from exp272 via queue_on_ckpt daemon 62026)
- Log: `/tmp/exp273.log` + `/hy-tmp/log/occluded_duke/exp273_psg3_t_od_s42/train_log.txt`
- Config: `configs/occluded_duke/prcv_best_tiny.yml` + CLI override (PSG_STAGES=`[-3,-2,-1]`)
- Scaffold: **Swin-Tiny + PSG stages [-3,-2,-1]** (LGPA/GCN/OA-SD/PLBOA/ParAug 全部关)
- 速度: ~99s/epoch,ETA 20:19 + 3h18min = ~23:37 CST

## 对照(Phase 3-A 矩阵)

| Exp | PSG stages | FINAL mAP/R1 |
|-----|-----------|-------------|
| exp270 | 无 | 59.2 / 68.4 |
| exp271 | `[-1]` | 60.2 / 69.5 |
| exp272 | `[-2,-1]` | **60.5 / 69.7** |
| **exp273 (本)** | `[-3,-2,-1]` | 进行中 e29/120 |

核心: **exp272 vs exp273 的 mAP 差就是增加 stage 1 PSG 的边际贡献**。历史预期 3-stage ≤ 2-stage。

## 进度

- 启动后 ~50min,**e29/120**,loss 从 1.0 下降到 0.83 (与 exp272 e29 早期轨迹吻合)
- id_global ~0.75, tri_global ~0.07,LR 6.9e-4 (ramp 阶段)
- queue daemon 62026 已退出 (任务完成),exp273 后无后继

## 自动化状态

- daemon 62026 已触发成功,exp273 独立运行中
- exp273 完成后 srvB 该 slot 空闲 → 可接 Phase 3-B 第一个 Tiny-GCN run
