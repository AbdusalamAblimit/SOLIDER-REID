# exp263b_best_b_od_s42_full120 monitor — Base Full Scaffold Occ-Duke seed 42 restart (lab4090)

- 机器: lab4090 (24GB 4090, mmpose-abu env, /home/afr/SOLIDER-REID/)
- 启动: 2026-04-22 00:24 CST (auto-chain restart, TEST.IMS_PER_BATCH 64)
- Log: `/home/afr/SOLIDER-REID/log/occluded_duke/exp263b_best_b_od_s42_full120/train_log.txt` (UTC clock)
- Config: `configs/occluded_duke/prcv_best_base.yml` + CLI `SOLVER.SEED 42 TEST.IMS_PER_BATCH 64`
- Scaffold: Swin-Base + Full Scaffold (LGPA + GCN512 + OA-SD + ParAug + PLBOA + 2-stage PSG `[-2,-1]`)
- Speed: ~4.2 min/epoch (4090, BS=64, 227 iter, TEST.IMS_PER_BATCH 64 避免 OOM)
- 总训练时长: 16h23min (2026-04-22 00:24 CST → 16:47 CST 2026-04-23)
- **动机**: exp263 seed 42 原 e80 eval OOM, 仅 e100 eff 72.5/81.8; 用户命令 "restart 跑满 120"

## 训练轨迹 (flip-test, eq_concat global; UTC log time)

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 52.5 | 62.6 | - | 83.6 |
| 20 | 63.3 | 73.8 | - | 89.1 |
| 30 | 68.2 | 78.6 | - | 90.7 |
| 40 | 69.8 | 80.2 | - | 91.1 |
| 50 | 71.9 | 81.1 | - | 92.0 |
| 60 | 71.6 | 80.1 (dip) | - | 92.2 |
| 70 | 73.1 | 82.4 | - | 92.3 |
| 80 | 73.3 | 81.5 | - | 92.6 |
| 90 | 73.4 | 81.5 | - | 92.2 |
| 100 | **73.6** (mAP peak) | 82.0 | - | 92.2 |
| 110 | 73.5 | 81.5 | - | 92.2 |
| **120 FINAL** | **73.5** | **81.5** | **90.2** | **92.3** |

## FINAL (2026-04-23 08:47:17 UTC = 16:47:17 CST)

- **mAP: 73.5%**, **Rank-1: 81.5%**, R5: 90.2%, R10: 92.3%
- **对照 exp263 s42 original e100 eff (OOM 前)**: 72.5/81.8 → Δ **+1.0 / -0.3**
- **对照 exp263d s41 FINAL (lab3090 pwrlim 280W)**: 74.1/83.3 → Δ **-0.6 / -1.8**
- **对照 exp285b Small OD s42 FINAL**: 73.8/83.8 → Δ **-0.3 / -2.3** (Base vs Small 同 s42, R1 显著弱)
- e100 mAP 峰值 73.6, e110/e120 轻微回落至 73.5 (0.1 mAP 抖动)

## 🏆 Base OD 矩阵最终定位

| Exp | 设备 | seed | Epoch | mAP / R1 |
|-----|------|------|-------|----------|
| exp263 | srvB | 42 | e100 eff (OOM) | 72.5 / 81.8 |
| **exp263b (本)** | **lab4090** | **42** | **e120 FINAL** | **73.5 / 81.5** |
| exp263d | lab3090 280W | 41 | e120 FINAL | **74.1 / 83.3** ← Base OD SOTA |

**结论**:
- **seed 41 > seed 42** (exp263d 74.1 > exp263b 73.5 / +0.6)
- **full 120 epoch > e100 eff** (exp263b 73.5 > exp263 72.5 / +1.0, 但仍不如 exp263d 74.1)
- 论文 **Base OD 主数字仍用 exp263d 74.1/83.3** (seed 41 最强)
- exp263b 作 **seed 42 full 120 复现数据点**, 佐证 restart 有效 + seed 42 天然弱

## MaxSim+flip eval 待办

- ckpt: `/home/afr/SOLIDER-REID/log/occluded_duke/exp263b_best_b_od_s42_full120/transformer_120.pth`
- 在 lab4090 本地跑 (网络不稳, 不适合 rsync 回 srvA)
- 对照: exp263 old e100 eff MaxSim 74.5/84.0, exp263d MaxSim 75.2/84.8

## lab4090 网络状态

- 2026-04-23 12:23 起 tailscale 100.94.229.1 持续断续
- ~1.5h 完全 unreachable, 之后间歇性可达
- 训练未受影响 (local disk IO 正常), 仅远程 log 监控不稳
- FINAL 信号最终通过 heartbeat 16:47 + 重试确认

## lab4090 idle 状态 (FINAL 后)

- 主训练进程结束 @ 16:47 CST
- 无 auto-chain 后续 (daemon 706372 已消费完)
- 待跑: MaxSim eval (on-lab4090) or 用户指派新任务
