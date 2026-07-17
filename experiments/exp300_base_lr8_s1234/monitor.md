# exp300_base_lr8_s1234 monitor — Base OD Full Scaffold seed 1234 (lab4090, mirror exp295 recipe)

- 机器: lab4090 (24GB 4090, mmpose-abu env)
- 启动: 2026-04-25 02:05 UTC (10:05 CST)
- Log: `/home/afr/SOLIDER-REID/log/occluded_duke/exp300_base_lr8_s1234/train_log.txt`
- Config: `configs/occluded_duke/prcv_best_base.yml` + CLI `SOLVER.SEED 1234 SOLVER.BASE_LR 0.0008 TEST.IMS_PER_BATCH 64`
- Scaffold: Swin-Base + Full Scaffold (LGPA + GCN512 + OA-SD + ParAug + PLBOA + 2-stage PSG `[-2,-1]`)
- Speed: 240s/epoch (~4 min) on 4090, total ~8h
- **动机**: Base OD SOTA 探索 — Small s1234 (exp295 75.2/85.4) 比 s42 (exp285b 74.7/84.8) 高 0.5 mAP, 测 Base 是否同规律 → 可能破 exp263d s41 SOTA 75.2/84.8

## 训练轨迹 (eq+flip; 从 train_log.txt 读取)

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 57.3 | 67.6 | 81.0 | 85.6 |
| 20 | 65.1 | 75.9 | 86.6 | 89.5 |
| 30 | 69.1 | 80.2 | 88.7 | 90.9 |
| 40 | 71.3 | 81.8 | 90.2 | 92.3 |
| 50 | 71.9 | 82.4 | 90.3 | 92.4 |
| 60 | 73.2 | 83.5 | 90.6 | 92.2 |
| 70 | 73.2 | 83.5 | 90.7 | 92.4 |
| 80 | 73.6 | 83.3 | 91.2 | 92.8 |
| 90 | 73.6 | 83.2 | 90.7 | 92.5 |
| 100 | 74.0 | 84.2 (peak R1) | 91.0 | 92.9 |
| 110 | 74.0 | 83.7 | 91.1 | 92.8 |
| **120 FINAL** | **74.0** | **83.8** | **91.1** | **93.0** |

## FINAL (2026-04-25 19:11 CST = 10:50 UTC)

- **eq+flip (e120)**: mAP **74.0%**, R1 **83.8%**, R5 91.1%, R10 93.0%
- **Global cosine+flip (e120)**: 73.9 / 83.9
- **MaxSim hybrid+flip (e120)**: **75.0 / 85.0**

### e100 ckpt MaxSim eval (R1 peak)

| Ckpt | Global cosine+flip | **MaxSim hybrid+flip** | vs exp263d 75.2/84.8 |
|------|--------------------|------------------------|----------------------|
| e100 | 73.7 / 83.8 | **75.0 / 85.2** | -0.2 / **+0.4** R1 best |
| e120 (FINAL) | 73.9 / 83.9 | 75.0 / 85.0 | -0.2 / +0.2 |

**e100 ckpt R1 比 e120 略好 +0.2** (e120 训练后期 R1 微 dip 84.2 → 83.7-83.8 plateau, ckpt e100 抓住 R1 peak)。

## 🎯 对照 exp263d (Base OD s41 LR8 SOTA reference)

| Metric | exp263d (s41 lab3090 280W) | **exp300 (s1234 lab4090)** | Δ |
|--------|----------------------------|----------------------------|----|
| eq+flip mAP/R1 | 74.1 / 83.3 | **74.0 / 83.8** | **-0.1 / +0.5** |
| MaxSim+flip mAP/R1 | 75.2 / 84.8 | **75.0 / 85.0** | **-0.2 / +0.2** |

**结论**:
- mAP 上 exp300 微低 0.1-0.2 (noise level), **未破 exp263d SOTA**
- **R1 上 exp300 +0.2-0.5 微超** (R1 维度 seed 1234 > seed 41)
- 整体非常接近 exp263d, 不同 seed 下浮动正常

## 与 exp296 (Base OD s41 LR8 lab4090 repro) 对比

| Exp | Seed | eq+flip | MaxSim+flip |
|-----|------|---------|-------------|
| exp263d | 41 (lab3090) | 74.1 / 83.3 | 75.2 / 84.8 |
| exp296 | 41 (lab4090) | 73.7 / 81.7 | 74.9 / 83.8 |
| **exp300** | **1234 (lab4090)** | **74.0 / 83.8** | **75.0 / 85.0** |

**lab4090 上的 seed 比较** (相同硬件):
- s41 (exp296): 73.7 / 81.7 — eq+flip
- s1234 (exp300): 74.0 / 83.8 — eq+flip
- Δ: **+0.3 mAP / +2.1 R1**, seed 1234 显著好

**lab4090 vs lab3090 同 seed 41 比较**:
- lab3090 (exp263d): 74.1 / 83.3
- lab4090 (exp296): 73.7 / 81.7
- Δ: **-0.4 / -1.6**, lab4090 系统性偏低

**最终结论**: **paper Base OD 主表保持用 exp263d 75.2/84.8** (lab3090 SOTA 数字), exp300 作 seed 1234 补充数据点 (展示 seed effect)。

## 训练曲线观察

- e10-e60 上升: 57.3 → 65.1 → 69.1 → 71.3 → 71.9 → 73.2 (健康 warm-up)
- e70-e90 plateau: 73.2 → 73.6 → 73.6 (稳定 plateau)
- e100 peak: 74.0 / 84.2 (R1 peak)
- e110-e120 plateau: 74.0 (mAP) / 83.7-83.8 (R1)

## lab4090 idle (FINAL 后)

- 主训练进程结束 @ 10:50 UTC (19:11 CST)
- MaxSim eval 完成 @ ~19:13 CST
- lab4090 可接下一任务
