# exp266b_best_b_op_s41 — Base OP seed 41 重跑 (5060Ti + TEST BATCH 128)

## 变体说明

- **exp266** (seed 42, srvC) e70 silent exit,e60 eff FINAL 78.4/86.2,与 exp265 Small 持平无增益
- 用户指示: srvA 回归 + OP SOTA 刷,exp266b seed 41 替代 exp266 给出完整 Base OP e120 FINAL
- **关键修改**: `TEST.IMS_PER_BATCH 256 → 128` 避免 5060Ti + Base eval OOM (历史 exp263/exp269 OOM 根因)

## 本 exp 变量

相对 exp266 (seed 42) 两变量:
1. `SOLVER.SEED` 42 → 41
2. `TEST.IMS_PER_BATCH` 256 → 128 (避免 eval OOM)

其他完全相同: Base backbone + Full Scaffold + PSG [-2,-1] + GCN512 + LGPA + OA-SD + ParAug + LOWER_BODY_OCC。

## CLI 配置

```bash
python3 train.py --config_file configs/occluded_posetrack/prcv_best_base.yml \
  SOLVER.SEED 41 \
  TEST.IMS_PER_BATCH 128 \
  OUTPUT_DIR /hy-tmp/log/occluded_posetrack/exp266b_best_b_op_s41
```

## 输出

- 机器: srvA (5060Ti 16G, auto-chain from exp265b via daemon 992)
- 预计时长: ~20-24h (Base OP on 5060Ti + WITH_CP, with_cp enabled in Base config)
- ETA: exp265b FINAL tmr 00:55 + 22h = 后天 22:00 CST

## 对照

| Exp | seed | 机器 | TEST BATCH | FINAL mAP/R1 |
|-----|------|------|------------|--------------|
| exp264 | 42 | srvC | 256 | 76.7/85.1 (Tiny) |
| exp265 | 42 | srvC | 256 | 78.4/86.2 (Small) |
| exp266 | 42 | srvC | 256 | e60 eff 78.4/86.2 (silent exit) |
| **exp265b** | **41** | **srvA** | 256 | pending |
| **exp266b (本)** | **41** | **srvA** | **128** | pending |

## 预期

- 若 exp266b FINAL > 78.5: Base 终于显示对 Small 的增益 (seed 42 可能锁死 Base=Small 假象)
- 若 exp266b ≈ exp265b: Base OP 确认 = Small OP (符合 exp266 初步 signal)
- FINAL 给出 e120 完整数字, 修复 exp266 "e60 eff FINAL" 的主表瑕疵

## paper 价值

- OP 补充 benchmark 完整性 (Tiny/Small/Base 三行均 e120 FINAL)
- seed 41 替代 seed 42 和 exp263d 同策略一致性
- TEST BATCH 128 降低成为 5060Ti + Base 默认配置
