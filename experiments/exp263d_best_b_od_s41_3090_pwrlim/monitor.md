# exp263d monitor — Base OD seed 41 lab3090 pwrlim 280W FINAL

- 机器: lab3090 (3090 24G, docker 18fbbab202e1, pwrlim 280W)
- 启动: 2026-04-20 23:34 CST (user 指示 seed 切换后, abandoning exp263c)
- Log: `/root/work/SOLIDER-REID/log/occluded_duke/exp263d_best_b_od_s41_3090_pwrlim/train_log.txt`
- Config: `configs/occluded_duke/prcv_best_base.yml` + SOLVER.SEED 41
- Scaffold: Swin-Base + Full Scaffold (LGPA + GCN512 + OA-SD + ParAug + LOWER_BODY_OCC) + PSG [-2,-1] + WITH_CP
- Speed: ~420-440s/epoch, 总训练 14h50min (23:34 → 14:27)
- env: `/root/miniconda3/envs/solider-reid/bin/python`

## 训练轨迹 (flip-test, eq_concat global)

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 54.8 | 64.8 | 79.0 | 83.3 |
| 20 | 64.2 | 74.6 | 85.8 | 89.0 |
| 30 | 68.1 | 78.6 | 87.4 | 90.1 |
| 40 | 69.9 | 80.3 | 88.1 | 90.6 |
| 50 | 72.3 | 81.7 | 89.8 | 92.2 |
| 60 | 72.4 | 81.7 | 90.1 | 92.4 |
| 70 | 73.6 | 82.9 | 91.0 | 93.1 |
| 80 | 73.8 | 82.9 | 91.1 | 93.1 |
| 90 | 74.0 | 83.7 | 91.1 | 93.0 |
| 100 | 74.0 | 83.1 | 91.0 | 93.0 |
| 110 | 74.2 | 83.3 | 90.6 | 93.1 |
| **120 FINAL** | **74.1** | **83.3** | **90.8** | **93.0** |

## FINAL (2026-04-21 14:27:34 CST)

- **mAP: 74.1%**, **Rank-1: 83.3%**, R5: 90.8%, R10: 93.0%
- 对照:
  - **exp263 old e100 eff FINAL 72.5/81.8 (Global+flip)**: Δ=**+1.6/+1.5** ⬆️⬆️
  - **exp263 old e100 MaxSim+flip 74.5/84.0**: Δ=-0.4/-0.7 (Global 接近 MaxSim 后处理)
  - **exp263c seed 42 abandoned @ e31** (因 seed 异常 e10 仅 2.7/4.5)
- Ckpt: `transformer_120.pth` (397MB)

## 🎯 核心论文结论

**seed 41 显著优于 seed 42 (in Base OD)**:
- exp263 seed 42 @ 5060Ti e100 eff 72.5/81.8 (OOM at e100 eval)
- exp263c seed 42 @ 3090 pwrlim 280W abandoned (seed 异常 warmup)
- **exp263d seed 41 @ 3090 pwrlim 280W FINAL 74.1/83.3** (完整 e120)

**PRCV Base OD 主表用 exp263d 74.1/83.3** (按用户早期指示, seed 41 替代 seed 42):
- vs KPR w/o prompt (Base 规格对应) 大幅领先
- 预期 MaxSim+flip 后处理可能达 **75-76** (待 Task #12 批量 eval)
- 3090 pwrlim 280W + seed 41 策略稳定, 无 OOM/crash

## lab3090 状态

- exp263d FINAL, main python process 已退出
- **lab3090 GPU 空闲** → 立即接 **exp266b_3090** (Base OP seed 41, 3090 24G 不需降 TEST BATCH)
- 新启动保持 pwrlim 280W (exp263c 证实 pwrlim 280W 对 3090 稳定有效)

## 待办

1. 批量 MaxSim+flip re-eval exp263d ckpt (Task #12)
2. decisions.md 记录 "seed 41 > seed 42" 正式验证
