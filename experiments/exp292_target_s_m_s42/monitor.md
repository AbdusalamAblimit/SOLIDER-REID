# exp292 monitor — Small Full Scaffold + target-heatmap on Market-1501 (seed 42)

- 机器: lab3090 (3090 24GB, solider-reid conda env, docker container 内, Market pose_data 本地)
- 启动: 2026-04-22 12:52 CST (restart-2 with TEST.IMS_PER_BATCH 64 after initial OOM @ e20 eval)
- Log: `/tmp/exp292.log` + `/root/work/SOLIDER-REID/log/market1501/exp292_target_s_m_s42/train_log.txt`
- Config: `configs/market/prcv_best_small.yml` + CLI `SOLVER.SEED 42 MODEL.POSE_USE_TARGET_HEATMAP True TEST.IMS_PER_BATCH 64`
- Scaffold: Swin-Small + Full Scaffold (LGPA + GCN512 + OA-SD + ParAug + **NO PLBOA** Market default) + 2-stage PSG + target-heatmap
- Speed: ~280-340s/epoch

## 训练轨迹

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 89.7 | 95.3 | - | 99.0 |
| 20 | 91.8 | 96.1 | - | 99.1 |
| 30 | 92.7 | 96.3 | 98.6 | 99.3 |
| 40 | 93.4 | 96.7 | 99.1 | 99.3 |
| 50 | 93.4 | 96.6 | 98.8 | 99.2 |
| 60 | 93.9 | 97.0 | 99.1 | 99.5 |
| 70 | 94.1 | 96.9 | 98.9 | 99.5 |
| 80 | 94.2 | 97.1 | 99.1 | 99.5 |
| 90 | 94.2 | 97.1 | 99.2 | 99.5 |
| **e90 eff FINAL** (训练停于 2026-04-22 23:25 CST, 用户请求让出 lab3090) | | | | |

## FINAL (effective, 训练在 e90 后停)

- **e90 effective FINAL: mAP 94.2%, R1 97.1%, R5 99.2%, R10 99.5%**
- 训练止于 e93 (log 显示 e93 iter 进行中) by 用户指令 `让出 lab3090`
- 对照 exp268 Small Market FINAL (scene) 94.3/97.3/99.1/99.5: Δ **-0.1 / -0.2 / +0.1 / 0** (near no-op, 符合 design.md 预期 Market 全 single-person target ≈ scene)

## 🎯 target-heatmap 三数据集收敛模式

| exp | dataset | 特性 | FINAL/eff | vs scene baseline | 结论 |
|-----|---------|------|-----------|-------------------|------|
| exp290 | Small OP | multi-person 多 | running e60 78.1/86.0 | vs exp265b -0.1/0 | near持平, 无 SOTA 推动 |
| exp291 | Small OD | mixed | FINAL 73.5/82.9 | vs exp285b -0.3/-0.9 | near no-op |
| **exp292** | **Small Market** | all single | **e90 eff 94.2/97.1** | vs exp268 **-0.1/-0.2** | **严格持平, 机制无回归** |

**target-heatmap 三数据集完整闭环**: **机制对 single-person 无回归, 对 multi-person 无显著 SOTA 加持**。原 design.md 核心假设 "OP 多人场景 SOTA" 未兑现, 改作 supplementary 消融 (证明机制 backward-compat + 数据集泛化)。

## 历史: OOM crash + restart

**Attempt 1** (2026-04-22 12:52 CST, TEST.IMS_PER_BATCH 默认 256):
- 训练到 e20, 进入 flip-test eval 时 OOM
- GPU: 24GB 24Gi, zombie 13.5GB + attempt 9.3GB → free 1.4GB 不够 494 MiB alloc
- Traceback: `attn = (q @ k.transpose(-2, -1))` OOM

**Attempt 2 (本记录)** @ 12:52 restart with `TEST.IMS_PER_BATCH 64`:
- e10-e90 全部 eval 成功, 无 OOM
- fix 生效

## 论文定位

- main_results Table 1 Small Market 行: 主数字仍用 exp268 94.3/97.3 (scene, FINAL 完整)
- exp292 作 target-heatmap 消融 supplementary: 证明 Market (全 single-person) 上机制 no-op, 和 OD (exp291) / OP (exp290) 结论一致
- target-heatmap 机制价值重新定位: **不是 SOTA 推动工具, 是可选 multi-person disambiguation (not needed for saturated benchmarks)**

## 停止原因

- 2026-04-22 23:25 CST 用户请求 "3090 有人用了, 停了吧"
- 已 pkill -9 -f exp292_target_s_m_s42
- ckpt transformer_80.pth (e80) 已保存 (latest 200M ckpt step 50-90 per CHECKPOINT_PERIOD=20)
