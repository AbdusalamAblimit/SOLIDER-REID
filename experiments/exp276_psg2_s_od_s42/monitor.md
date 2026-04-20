# exp276 monitor — Phase 3-A Small 2-stage PSG (Occ-Duke, seed 42)

- 机器: lab4090
- 启动: 2026-04-20 23:37 CST (auto-chain from exp275 via v2 daemon 3654950)
- Log: `/home/afr/SOLIDER-REID/log/occluded_duke/exp276_psg2_s_od_s42/train_log.txt`
- Config: `configs/occluded_duke/prcv_best_small.yml` + CLI override (PSG_STAGES=`[-2,-1]`)
- Scaffold: Swin-Small + PSG 2-stage (LGPA/GCN/OA-SD/PLBOA/ParAug 全关)
- Speed: ~54s/epoch, 总训练 ~1h50min

## 对照 (Phase 3-A Small)

| Exp | PSG stages | FINAL mAP/R1 |
|-----|-----------|-------------|
| exp274 | 无 | 68.1 / 76.8 |
| exp275 | `[-1]` | 68.8 / 76.8 |
| **exp276 (本)** | `[-2,-1]` | **68.3 / 77.2** |
| exp277 | `[-3,-2,-1]` | 49.0 / 57.7 (塌缩, 见 exp277 monitor) |

## 训练轨迹 (flip-test, eq_concat global)

| Epoch | mAP | R1 | R5 | R10 | 备注 |
|-------|-----|----|----|----|------|
| 10 | 43.5 | 52.6 | 66.9 | 72.5 | 比 exp274/275 同期低 warmup delay |
| 20 | 42.2 | 49.8 | 65.2 | 71.4 | ⚠️ 从 e10 微降 |
| 30 | 57.9 | 65.6 | 79.0 | 84.3 | **大幅恢复 +15.7/+15.8** |
| 40 | 62.4 | 70.2 | 83.0 | 86.8 | |
| 50 | 63.4 | 71.6 | 83.6 | 87.2 | 超 exp275 e50 62.6 (+0.8) |
| 60 | 66.4 | 75.2 | 85.3 | 88.8 | |
| 70 | 66.1 | 74.8 | 85.6 | 88.9 | 持平 exp275 |
| 80 | 67.8 | 77.1 | 87.2 | 90.0 | **R1 超 exp275 +1.4** |
| 90 | 67.9 | 77.1 | 87.0 | 90.1 | |
| 100 | 68.0 | 76.7 | 87.2 | 90.2 | |
| 110 | 68.2 | 76.8 | 87.1 | 90.1 | |
| **120 FINAL** | **68.3** | **77.2** | **87.2** | **90.1** | |

## FINAL (2026-04-21 01:41 CST)

- **mAP: 68.3%**, **Rank-1: 77.2%**, R5: 87.2%, R10: 90.1%
- 对照:
  - exp274 no-PSG 68.1/76.8 → Δ=**+0.2/+0.4**
  - exp275 1-stage 68.8/76.8 → Δ=**-0.5 mAP / +0.4 R1** (2-stage mAP 比 1-stage 低!)
- Ckpt: `transformer_120.pth` (200MB)

## 结论 (Small 2-stage vs 1-stage 异常)

Small 上 2-stage PSG 相比 1-stage:
- **mAP 下降 0.5** (68.8 → 68.3)
- **R1 上升 0.4** (76.8 → 77.2)

与 Tiny 上 2-stage > 1-stage (+0.3 mAP/+0.2 R1) 模式不同。Small 上 2-stage 可能达到 "**整体稳健性-top1 准确性**" trade-off 的不同点,mAP 和 R1 反向。

中期 (e20 42.2 → e30 57.9) **+15.7 恢复** 也表明 2-stage PSG 在 Small 早期 warmup 更曲折。

**Phase 3-A Small 矩阵到此**:
- exp274 (no-PSG): 68.1/76.8
- exp275 (1-stage): **68.8**/76.8 (mAP peak)
- exp276 (2-stage): 68.3/**77.2** (R1 peak)
- exp277 (3-stage): **49.0/57.7 塌缩** (见 exp277 monitor)

论文 Table 2 写法: "Small 上 PSG stage 收益模式与 Tiny 不同,1-stage 达 mAP peak, 2-stage R1 peak, 3-stage 训练塌缩"。

auto-chain → exp277 Small 3-stage (daemon 3654948)
