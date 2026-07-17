# exp269b_best_b_m_s42_full120 monitor — Base Full Scaffold Market-1501 seed 42 restart full 120

- 机器: srvC (5060Ti 16G, i-2:25551)
- 启动: 2026-04-23 05:41 CST (auto-chain restart from exp289 FINAL)
- Log: `/hy-tmp/log/market/exp269b_best_b_m_s42_full120/train_log.txt`
- Config: `configs/market/prcv_best_base.yml` + CLI `SOLVER.SEED 42 TEST.IMS_PER_BATCH 64`
- Scaffold: Swin-Base + Full Scaffold (LGPA + GCN512 + OA-SD + ParAug + PLBOA **OFF** + 2-stage PSG `[-2,-1]`)
- Speed: ~9.4 min/epoch (5060Ti + BS=64, 186 iter, TEST.IMS_PER_BATCH 64 避免 OOM)
- 总训练时长: 19h36min (05:41 → 01:17 tmr)
- **动机**: exp269 s42 原 e80 eval OOM, 仅 e80 eff 94.4/97.0; 用户命令 "restart 跑满 120"

## 训练轨迹 (flip-test, eq_concat global)

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 87.6 | 94.5 | - | 99.0 |
| 20 | 91.4 | 95.7 | - | 99.1 |
| 30 | 92.7 | 96.5 | - | 99.1 |
| 40 | 93.5 | 96.8 | - | 99.2 |
| 50 | 93.9 | 97.1 | - | 99.2 |
| 60 | 94.1 | 97.0 | - | 99.3 |
| 70 | 94.4 | 97.1 | - | 99.4 |
| 80 | 94.4 | 97.1 | - | 99.4 |
| 90 | 94.5 | 97.2 | 99.0 | 99.4 |
| 100 | 94.5 | 97.2 | 99.0 | 99.4 |
| 110 | 94.5 | 97.1 | 99.0 | 99.5 |
| **120 FINAL** | **94.5** | **97.2** | **99.1** | **99.5** |

## FINAL (2026-04-24 01:17:24 CST)

- **mAP: 94.5%**, **Rank-1: 97.2%**, R5: 99.1%, R10: 99.5%
- **对照 exp269 s42 original e80 eff (OOM 前)**: 94.4/97.0/98.9/99.4 → Δ **+0.1/+0.2/+0.2/+0.1** (full 120 全面微优)
- **对照 exp269 MaxSim+flip** 94.5/97.1: eq_concat 本次即持平 MaxSim
- **对照 exp268 Small FINAL**: 94.3/97.3/99.1/99.5 → Δ **+0.2/-0.1/0/0** (Base vs Small 仅 mAP 0.2 优, R1 微弱)
- **对照 exp293 Base PLBOA ON**: 93.8/97.2 → Δ **+0.7/0** (**PLBOA 净 -0.7 mAP 代价 re-confirmed**)
- e90 开始 mAP 稳定 94.5 (30 epoch 平稳), R1 97.1-97.2 微幅抖动
- Ckpt: `/hy-tmp/log/market/exp269b_best_b_m_s42_full120/transformer_120.pth` (407MB)

## 🏆 Market Base 矩阵最终定位

| Exp | PLBOA | Epoch | mAP/R1 (Global+flip) | mAP/R1 (MaxSim+flip) |
|-----|-------|-------|---------------------|---------------------|
| exp269 orig | OFF | e80 eff (OOM) | 94.4/97.0 | 94.5/97.1 |
| **exp269b (本)** | **OFF** | **e120 FINAL** | **94.5/97.2** | **pending** |
| exp293 | ON | e80 eff (OOM, 后 restart 93.8) | 94.1/96.9 | 94.1/97.2 |
| exp293b restart | ON | e120 FINAL | 93.8/97.2 | pending |

**Market PLBOA 定位**: **net negative -0.7 mAP** (exp269b 94.5 vs exp293b 93.8), 论文主表 **Market Base 用 exp269b 94.5/97.2**。

## 跨域 Occluded-ReID eval 待办

exp269b PLBOA OFF Market → Occ-REID 跨域:
- exp269 orig e80 eff cross-domain: Global 85.0/89.0, MaxSim 88.2/91.2 (**Top-tier**)
- exp269b e120 ckpt 可能进一步提升 (训练更稳定), 预期 ≥ exp269 orig

## MaxSim+flip eval FINAL (srvC, 2026-04-24 01:25 CST)

| 评测模式 | mAP | R1 |
|---------|-----|----|
| train-side eq_concat+flip (FINAL) | 94.5 | 97.2 |
| Global cosine+flip | 94.4 | 97.1 |
| **MaxSim hybrid+flip** | **94.6** | **97.2** |

**MaxSim 增益**: +0.1 mAP / 0 R1 (vs eq_concat)

**对照 Market Base MaxSim 矩阵**:
| Exp | PLBOA | eq+flip | Global+flip | MaxSim+flip |
|-----|-------|---------|-------------|-------------|
| exp269 orig e80 eff | OFF | 94.4/97.0 | ~94.4 | 94.5/97.1 |
| **exp269b full 120** | **OFF** | **94.5/97.2** | 94.4/97.1 | **94.6/97.2** |
| exp293 e80 eff | ON | 94.1/96.9 | - | 94.1/97.2 |
| exp293b restart | ON | 93.8/97.2 | - | pending |

**结论**: exp269b MaxSim 94.6/97.2 **超 exp269 orig MaxSim 94.5/97.1** by +0.1/+0.1。
**论文 Market Base MaxSim 主数字升级为 94.6/97.2**。

## srvC idle (FINAL 后)

- 主进程结束 @ 01:17 CST
- 无 auto-chain 下游 (daemon 已消费完)
- 下一任务: srvC 本地 MaxSim eval + 可能跨域 eval
