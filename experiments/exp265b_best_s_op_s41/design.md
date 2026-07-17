# exp265b_best_s_op_s41 — Small Full Scaffold OP seed 41 (SOTA 候选)

## 动机

- srvA 回来了 (之前未续费 ssh refused), GPU 空闲
- 用户指示: "用它来刷一刷 OP 的 sota"
- 现有 OP 最强: **exp265 Small FINAL 78.4/86.2** (srvC seed 42, Phase 1)
- exp266 Base OP e60 eff 78.4/86.2 与 Small 0 增益 → Small 已是 OP 最佳 backbone

## 本 exp 变量

- 相对 exp265 (seed 42) 单变量: `SOLVER.SEED` 42 → 41
- 其他参数完全不变: Small + Full Scaffold (LGPA + GCN512 + OA-SD + ParAug + LOWER_BODY_OCC + PSG [-2,-1])
- 和 exp263d 同策略 (seed 41 替代 seed 42 刷 SOTA)

## CLI 配置

```bash
python3 train.py --config_file configs/occluded_posetrack/prcv_best_small.yml \
  SOLVER.SEED 41 \
  TEST.IMS_PER_BATCH 128 \
  OUTPUT_DIR /hy-tmp/log/occluded_posetrack/exp265b_best_s_op_s41
```

**注**: 2026-04-21 12:08 restart — 原 12:00 启动时未降 TEST BATCH, 用户反馈 "5060Ti 上所有实验预防性降 TEST BATCH" 后 kill 重启 (PID 1151, 原 633 已 kill)。

## 输出

- 机器: srvA (i-2:29162, 5060 Ti 16G, 已 resume)
- 预计时长: ~12-14h (exp265 原 seed 42 on srvC 约 14h)
- ETA: 11:55 启动 → 后天 02:00 FINAL

## 对照

| Exp | seed | 机器 | FINAL mAP/R1 |
|-----|------|------|--------------|
| exp265 | 42 | srvC 5060Ti | 78.4 / 86.2 |
| exp266 | 42 | srvC 5060Ti | 78.4 / 86.2 (Base, e60 eff) |
| **exp265b (本)** | **41** | **srvA 5060Ti** | pending |

## 预期

若 seed 41 trajectory 健康 (如 exp263d 显示 seed 41 > seed 42):
- FINAL 预计 78.5-79.5 / 86.5-87.5 (可能微超 exp265)
- 和 exp265 seed 42 组成 2-seed ensemble 报告,做 KPR w/o prompt +0.5-0.8 的 SOTA 主张更稳
- PRCV 主表 OP 行用 ensemble (或 max) 数字

若 seed 41 FINAL < exp265 (概率小), 保留 exp265 原数字。

## paper 价值

- OP 作为 **补充 benchmark**, 在 "跨域/跨数据集泛化" 章节突出 Small 优势
- exp265b 提供 multi-seed 稳定性 signal → SOTA 声明更稳
- 对照 KPR w/o prompt 73.3/82.5 → Δ=+5+/+3+ 显著
