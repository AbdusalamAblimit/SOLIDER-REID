# exp263d_best_b_od_s41_3090_pwrlim — Base OD Full Scaffold 重跑 (seed 41, pwrlim 280W)

## 变体说明

exp263 系列变体历史:
- **exp263**: 原 Base OD Full Scaffold seed 42,lab3090 跑 → e100 eff FINAL 72.5/81.8 (OOM kill)
- **exp263c**: lab3090 GPU hang 恢复后 restart seed 42 with pwrlim 280W → e10 2.7 / e20 17.0 / e30 39.0 (恢复缓慢,seed 42 轨迹异常)
- **exp263d (本)**: **seed 42 → seed 41** + 保持 pwrlim 280W 其他不变

## 动机

exp263c e10 mAP 2.7 / R1 4.5 异常低 (Base backbone 正常 e10 约 20+%)。e20 恢复到 17.0 但 trajectory 仍慢于 exp263 原 run。用户判断 **seed 42 可能有问题**,换 seed 41 重新验证。

按用户指示: "报告时就报告这个是 seed 41 就行" — 作为 exp263 系列代表的 FINAL 用 seed 41。

## 本 exp 变量

- 相对 exp263c (seed 42): 仅 `SOLVER.SEED` 42 → 41
- 其他: 同 exp263c (Base backbone + Full Scaffold + pwrlim 280W + docker 容器内 solider-reid env)

## CLI 配置

```bash
/root/miniconda3/envs/solider-reid/bin/python train.py \
  --config_file configs/occluded_duke/prcv_best_base.yml \
  SOLVER.SEED 41 \
  OUTPUT_DIR ./log/occluded_duke/exp263d_best_b_od_s41_3090_pwrlim
```

## 输出

- 机器: lab3090 (docker 容器 18fbbab202e1, 3090 24G, pwrlim 280W)
- Log: `/root/work/SOLIDER-REID/log/occluded_duke/exp263d_best_b_od_s41_3090_pwrlim/train_log.txt`
- 预计时长: ~14h (3090 + Base Full Scaffold + pwrlim 280W)
- 启动时间: 2026-04-20 23:30 CST

## 对照

- exp263 原 (seed 42 未 pwrlim,OOM): e100 eff FINAL 72.5/81.8
- exp263c (seed 42 pwrlim abandoned @ e31)
- exp263d 本 (seed 41 pwrlim): 预期 e120 FINAL ≥ 72.5/81.8
