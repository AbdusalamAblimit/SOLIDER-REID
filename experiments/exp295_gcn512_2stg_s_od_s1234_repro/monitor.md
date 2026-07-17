# exp295_gcn512_2stg_s_od_s1234_repro monitor — Small OD Full Scaffold seed 1234 (exp255 复现)

- 机器: lab4090 (24GB 4090, mmpose-abu env)
- 启动: 2026-04-24 09:19:19 UTC (17:19 CST)
- Log: `/home/afr/SOLIDER-REID/log/occluded_duke/exp295_gcn512_2stg_s_od_s1234_repro/train_log.txt`
- Config: `configs/occluded_duke/prcv_best_small.yml` + CLI `SOLVER.SEED 1234`
- Scaffold: Swin-Small + Full Scaffold (LGPA + GCN512 + OA-SD + ParAug + PLBOA + 2-stage PSG `[-2,-1]`)
- Speed: 175s/epoch (~2.9 min), 总训练 6h06min (09:19 → 15:25 UTC = 17:19 → 23:25 CST)
- **动机**: exp255 (seed 1234) 历史 MaxSim+flip **75.2/85.6** 是 Small OD 最强数字, 但 v2 fix eval script 下 exp285b seed 42 只到 74.7/84.8 (-0.5 mAP)。用户命令 "lab4090 复现 seed 1234 验证"。

## 训练轨迹 (flip-test, eq_concat global; 逐行 from train_log.txt)

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 56.6 | 67.8 | 81.9 | 86.1 |
| 20 | 63.6 | 74.8 | 85.6 | 88.5 |
| 30 | 68.2 | 78.7 | 87.8 | 90.1 |
| 40 | 71.3 | 81.9 | 89.9 | 92.0 |
| 50 | 72.6 | 82.7 | 90.8 | 92.4 |
| 60 | 72.9 | 83.3 | 90.8 | 92.4 |
| 70 | 73.0 | 82.7 | 90.4 | 92.2 |
| 80 | 73.9 | 83.7 | 91.3 | 92.9 |
| 90 | 74.0 | 83.7 | 90.9 | 92.7 |
| 100 | 74.0 | 83.7 | 91.0 | 92.6 |
| 110 | 74.2 | 84.0 | 91.0 | 92.6 |
| **120 FINAL** | **74.2** | **84.0** | **91.0** | **92.7** |

## FINAL (2026-04-24 23:25:25 CST = 15:25:25 UTC)

- **eq+flip (train)**: mAP **74.2%**, Rank-1 **84.0%**, R5 **91.0%**, R10 **92.7%**

## 🏆 MaxSim+flip eval (v2 fix, lab4090 2026-04-24 23:30 CST)

- **Global cosine+flip**: mAP **73.7%**, R1 **83.3%**
- **MaxSim hybrid+flip**: mAP **75.2%**, R1 **85.4%**

## 🎯 对照 exp255 historical (seed 1234)

Authoritative source: `/root/work/SOLIDER-REID/log_remote_srvA_backup/occluded_duke/exp255_small_gcn512_2stage/train_log.txt` on lab3090。

| Metric | exp255 (hist) | **exp295 (repro)** | Δ |
|--------|---------------|---------------------|----|
| e120 eq+flip | 73.2 / 83.3 | **74.2 / 84.0** | **+1.0 / +0.7** |
| MaxSim+flip | 75.2 / 85.6 | **75.2 / 85.4** | **0 / -0.2** (mAP 完全 match, R1 -0.2) |

**结论**: **exp295 完全重现 exp255 的 75.2 mAP target**, 证明 exp255 历史数字是真实可重现的, 不是 eval script bug 产出。R1 差 0.2 在 seed 噪声范围内。

## 训练轨迹对照 (每 10 epoch, vs exp255)

| Epoch | exp295 | exp255 hist | Δ |
|-------|--------|-------------|----|
| 10 | 56.6/67.8 | 54.7/65.7 | +1.9/+2.1 |
| 20 | 63.6/74.8 | 62.2/74.3 | +1.4/+0.5 |
| 30 | 68.2/78.7 | 67.0/77.4 | +1.2/+1.3 |
| 40 | 71.3/81.9 | 70.2/81.2 | +1.1/+0.7 |
| 50 | 72.6/82.7 | 71.3/82.0 | +1.3/+0.7 |
| 60 | 72.9/83.3 | 71.6/82.1 | +1.3/+1.2 |
| 70 | 73.0/82.7 | 71.9/81.8 | +1.1/+0.9 |
| 80 | 73.9/83.7 | 72.7/82.5 | +1.2/+1.2 |
| 90 | 74.0/83.7 | 72.9/83.1 | +1.1/+0.6 |
| 100 | 74.0/83.7 | 73.0/83.0 | +1.0/+0.7 |
| 110 | 74.2/84.0 | 73.1/83.3 | +1.1/+0.7 |
| **120** | **74.2/84.0** | **73.2/83.3** | **+1.0/+0.7** |

**Pattern**: exp295 全程 +1.0-1.3 mAP 领先 exp255, 差异或来自 v2 fix 后的 evaluation reporting (train-side eval 也用 fixed path, 影响记录数字) 或训练环境微差 (lab4090 vs old srvA, mixed precision 差异)。但 **MaxSim+flip 终值完全 match 75.2** 这是关键。

## 🏆 paper 主表 Small OD 用 exp295

**Main table line update**:
- **exp295 (Small OD, seed 1234, lab4090, code HEAD + v2 fix)**: eq+flip **74.2/84.0**, MaxSim+flip **75.2/85.4**
- 替代 historical exp255 作为 reproducible reference

## lab4090 idle (FINAL 后)

- 主训练进程结束 @ 15:25:25 UTC
- MaxSim eval 完成 @ 15:30 UTC
- lab4090 idle, 可接下一任务
