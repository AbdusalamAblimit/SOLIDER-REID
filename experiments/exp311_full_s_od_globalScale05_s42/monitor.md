# exp311 / exp311b — Small OD Full Scaffold + 真生效 GLOBAL_LOSS_SCALE=0.5

## 背景
config 中 `MODEL.GLOBAL_LOSS_SCALE: 0.5` 早期设置但代码 bug 只在 no-part 路径生效, Full Scaffold 走 part-path 完全忽略。本次 commit `c059dca` 修复, 让 0.5 真在 part-path 生效 → global ID/Tri 各乘 0.5 (global=0.25 part=0.5)。

## 实验记录

### exp311 s42 (KILLED)
- 启动: 2026-04-27 10:54 (lab4090)
- e10 mAP: **19.7%** (vs baseline 70%) — 严重 underfit, seed 42 不利
- 决定 kill 切 s1234 重启

### exp311b s1234 (effective FINAL @ e100)
- 启动: 2026-04-27 11:47 (lab4090)
- e100 OOM crash @ e101 (同学 gaitheat 进程回来抢 GPU 16GB)
- e100 ckpt 已保存为 effective FINAL

| Epoch | mAP | R1 | Δ vs exp295 (无 scale) |
|-------|-----|-----|----------------------|
| 10 | 54.2 | 66.2 | (no scale e10 ~ 70+) |
| 20 | 61.4 | 72.5 | -6.8 / -6.2 |
| 30 | 66.5 | 77.0 | -4.8 / -4.9 |
| 40 | 69.5 | 79.5 | -3.1 / -3.2 |
| 50 | 71.7 | 81.6 | -1.2 / -1.7 |
| 60 | 71.9 | 81.8 | -1.1 / -0.9 |
| 70 | 72.5 | 82.4 | -1.4 / -1.3 |
| 80 | 73.2 | 83.3 | -0.8 / -0.4 |
| 90 | 73.4 | 82.7 | -0.6 / -1.0 |
| **100 effective FINAL** | **73.5** | **83.2** | **-0.7 / -0.8** |

### MaxSim eval (e100 ckpt)
- eq+flip: **73.5 / 83.2**
- Global cosine+flip: 72.7 / 82.2
- **MaxSim hybrid+flip: 74.5 / 84.8**

## 对照 — exp295 s1234 baseline (无 scale, GLOBAL_LOSS_SCALE 实际 = 1.0)

| 指标 | exp311b (0.5× scale 真生效) | exp295 (无 scale) | Δ |
|------|----------------------------|---------------------|----|
| eq+flip | 73.5/83.2 | 74.2/84.0 | -0.7/-0.8 |
| Global+flip | 72.7/82.2 | 73.7/83.3 | -1.0/-1.1 |
| **MaxSim** | **74.5/84.8** | **75.2/85.4** | **-0.7/-0.6** |

## 结论

**GLOBAL_LOSS_SCALE=0.5 真生效后净负 -0.7 mAP / -0.6 R1 MaxSim**, 即使 e120 跑完估计也只能补 +0.1-0.3 (cosine 末段), 仍 < baseline。

**Paper claim**: 0.5× global loss scale **不是有效改进**, baseline (effective 1.0) 更好。

## 后续动作

- prcv_best_*.yml 配置里的 `GLOBAL_LOSS_SCALE: 0.5` 应改回 1.0 或显式 CLI override 1.0 来保持向后一致性
- 启动 Tiny 5-way 损失权重 sweep (exp312-316), 探索其他 weight 维度
