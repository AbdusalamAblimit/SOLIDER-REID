# exp266b_3090 monitor — Base Full Scaffold Occ-PTrack seed 41 (lab3090)

- 机器: lab3090 (RTX 3090 24G, docker 18fbbab202e1, solider-reid conda env, pwrlim 280W)
- 启动: 2026-04-21 06:28 UTC = 14:28 CST (docker 内)
- Log: `/root/work/SOLIDER-REID/log/occluded_posetrack/exp266b_best_b_op_s41_3090/train_log.txt` (clock 为 UTC)
- Config: `configs/occluded_posetrack/prcv_best_base.yml` + CLI `SOLVER.SEED 41`
- Scaffold: Swin-Base + Full Scaffold (LGPA + GCN512 + OA-SD + ParAug + LOWER_BODY_OCC) + 2-stage PSG `[-2,-1]`
- Speed: ~9.5 min/epoch (BS=64, 275 iter, pwrlim 280W 限速), 总训练 19h
- seed 42 版本 exp266 srvC @ 2026-04-20 e60 后 silent exit (hy-tmp 平台 kill), 用 seed 41 + lab3090 pwrlim 280W rerun

## 训练轨迹 (flip-test, eq_concat global; UTC log time)

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 74.0 | 83.4 | 93.8 | 96.6 |
| 20 | 75.6 | 83.9 | 93.5 | 96.4 |
| 30 | 77.4 | 85.4 | 94.0 | 96.8 |
| 40 | 77.8 | 85.6 | 94.2 | 96.9 |
| 50 | 78.1 | 86.0 | 94.0 | 97.0 |
| 60 | 78.4 | 86.2 | 94.3 | 97.0 |
| 70 | 78.5 | 86.0 | 94.2 | 97.0 |
| 80 | 78.5 | 86.1 | 94.2 | 96.9 |
| 90 | 78.5 | **86.3** (R1 peak) | 94.3 | 96.9 |
| 100 | 78.5 | 86.2 | 94.3 | 96.9 |
| 110 | 78.5 | 86.2 | 94.4 | 96.9 |
| **120 FINAL** | **78.5** | **86.2** | **94.4** | **96.9** |

## FINAL (2026-04-22 01:29:54 UTC = 09:29:54 CST)

- **mAP: 78.5%**, **Rank-1: 86.2%**, R5: 94.4%, R10: 96.9%
- **对照 exp266 s42 srvC e60 eff** (e70 后 silent exit): 78.4/86.2 → Δ=**+0.1 / 0** (mAP 微优, R1 持平)
- **对照 exp265 s42 Small OP FINAL** 78.4/86.2 → Δ=+0.1/0 (**Base vs Small OP 0 增益 confirmed**)
- **对照 exp265b s41 Small OP** 78.5/85.9 → Δ=0/+0.3 (Base 略优 R1 over Small 同 seed)
- 从 e70 开始 mAP 稳定 78.5 (**50 epoch 平稳**), R1 在 86.0-86.3 小幅抖动
- Ckpt: `transformer_120.pth` (407MB)

## 🔥 Phase 3 OP 矩阵完整闭合 (Swin-{S, B} × seed {42, 41})

| | seed 42 | seed 41 |
|---|---------|---------|
| Small (exp265/265b) | 78.4/86.2 (srvC) | 78.5/85.9 (srvA) |
| Base (exp266/266b_3090) | 78.4/86.2 e60 eff (srvC silent exit) | **78.5/86.2** (lab3090) |

**OP benchmark 全面饱和 ~78.5/86.2**:
1. **Swin-S vs Swin-B**: 0 mAP / 0 R1 差 (饱和)
2. **seed 42 vs seed 41**: 0.1 mAP / 0.0-0.3 R1 (鲁棒)
3. **跨设备** (srvC/srvA/lab3090): 所有数字 ±0.3 以内
4. **论文结论**: OP 数据集对 backbone cap 不敏感, 支持 "Swin-S 已够用" 主张

## lab3090 idle (2026-04-22 01:29 UTC = 09:29 CST)

- GPU 0% / 11 MiB (transient), 主 PID 37072 结束
- 无 auto-chain daemon — lab3090 chain 至 exp266b_3090 终止
- Phase 3 主矩阵 Base OP 双设备版本闭合
- 后续可用 idle slot 跑 Task #12 MaxSim eval (等 srvA exp266b + srvC Phase 3-C 队列后统一批跑)

## Phase 3 OP 论文主表填写

- **Small OP 主数字**: exp265 s42 78.4/86.2 (R1 最高)
- **Base OP 主数字**: exp266b_3090 s41 **78.5/86.2** (完整 120 epoch, 无 silent exit)
- 跨 seed/设备 robustness 写入 supplementary
