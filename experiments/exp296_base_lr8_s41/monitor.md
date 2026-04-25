# exp296_base_lr8_s41 monitor — Base OD Full Scaffold seed 41 LR 8e-4 (lab4090 repro of exp263d)

- 机器: lab4090 (24GB 4090, mmpose-abu env)
- 启动: 2026-04-24 23:42 CST
- Log: `/home/afr/SOLIDER-REID/log/occluded_duke/exp296_base_lr8_s41/train_log.txt`
- Config: `configs/occluded_duke/prcv_best_base.yml` + CLI `SOLVER.SEED 41 SOLVER.BASE_LR 0.0008 TEST.IMS_PER_BATCH 64`
- Scaffold: Swin-Base + Full Scaffold (LGPA + GCN512 + OA-SD + ParAug + PLBOA + 2-stage PSG `[-2,-1]`)
- Speed: 241s/epoch (~4 min) on 4090, total ~8h
- **动机**: overnight Base OD LR sweep, LR 8e-4 baseline = repro exp263d 75.2/84.8 (with current code + v2 fix eval)

## 训练轨迹 (eq+flip; 从 train_log.txt 读取)

| Epoch | mAP | R1 |
|-------|-----|----|
| 10 | 48.3 | 57.8 |
| 20 | 63.6 | 75.0 |
| 30 | 68.2 | 78.8 |
| 40 | 70.4 | 80.2 |
| 50 | 72.2 | 81.5 |
| 60 | 71.8 | (dip) |
| 70 | 73.2 | — |
| 80 | 73.5 | — |
| 90 | 73.5 | — |
| 100 | 73.5 | — |
| 110 | 73.7 | — |
| **120 FINAL** | **73.7** | **81.7** |

## FINAL (2026-04-25 07:58:33 CST = 23:58:33 UTC)

- **eq+flip**: mAP **73.7%**, R1 **81.7%**, R5 **90.0%**, R10 **92.5%**
- **Global cosine+flip**: 72.6 / 81.0
- **MaxSim hybrid+flip**: **74.9 / 83.8**

## 🎯 对照 exp263d (Base OD s41 LR8 reference)

| Metric | exp263d (lab3090 pwrlim 280W) | **exp296 (lab4090)** | Δ |
|--------|-------------------------------|----------------------|----|
| eq+flip mAP/R1 | 74.1/83.3 | **73.7/81.7** | -0.4/-1.6 |
| MaxSim+flip mAP/R1 | 75.2/84.8 | **74.9/83.8** | -0.3/-1.0 |

**结论**: exp296 reproducibility 接近但不完美。R1 系统性偏低 ~1.0-1.6 mAP, mAP 偏低 ~0.3-0.4。差异可能来自:
1. lab4090 4090 vs lab3090 3090 (不同硬件/编译/数值)
2. 训练 seed-state 差异 (不同 GPU 上 cuda RNG 不一)
3. lab3090 280W 限功率可能改变 mixed-precision 行为

**Paper 主表 Base OD 维持 exp263d 75.2/84.8** (lab3090 ckpt 真实 SOTA), exp296 作 reproducibility 注脚。

## 训练曲线观察

- e10-e50 标准 warmup → linear climb (48 → 72)
- e60 dip 0.4 mAP (从 72.2 → 71.8) — 单点 noise
- e70-e120 plateau 73.5-73.7 (低于 exp263d 73.9-74.1 同期)
- 整体收敛健康, 无 NaN/Inf

## lab4090 idle (FINAL 后)

- 主训练进程结束 @ 23:58:33 UTC
- MaxSim eval 完成 @ 00:01 UTC (08:01 CST)
- lab4090 现 idle, 可接下一任务
