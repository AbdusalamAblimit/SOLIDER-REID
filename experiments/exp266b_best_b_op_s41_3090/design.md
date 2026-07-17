# exp266b_best_b_op_s41_3090 — Base OP seed 41 on lab3090 (3090 24G)

## 变体说明

- exp263d Base OD seed 41 FINAL @ 14:27 (74.1/83.3, 超 exp263 old +1.6/+1.5)
- lab3090 空闲 → 立即接 Base OP seed 41 重跑 (修 exp266 silent exit 瑕疵, 和 exp263d 同策略)
- 本 exp 在 lab3090 (3090 24G pwrlim 280W), 不需要降 TEST BATCH (vs exp266b on srvA 5060Ti 16G 需要 128)

## 本 exp 变量

相对 exp266 (seed 42, srvC silent exit e70) 单变量: `SOLVER.SEED` 42 → 41
保持 Base Full Scaffold + PSG [-2,-1] + GCN512 + LGPA + OA-SD + ParAug + LOWER_BODY_OCC + WITH_CP.

**和 srvA daemon 992 触发的 exp266b 的关系**:
- srvA exp266b (daemon 992 触发, 等 exp265b FINAL, ETA tmr 06:10+24h): seed 41, 5060Ti 16G, TEST BATCH 128
- **本 exp266b_3090** (立即启动 @ 14:29): seed 41, 3090 24G, TEST BATCH 256 default
- 双 run 同 seed 不同设备, 组成跨设备验证 + 如都 FINAL 则取更稳的 (3090 更快, 5060Ti 提供对照)

## CLI 配置

```bash
/root/miniconda3/envs/solider-reid/bin/python train.py \
  --config_file configs/occluded_posetrack/prcv_best_base.yml \
  SOLVER.SEED 41 \
  OUTPUT_DIR ./log/occluded_posetrack/exp266b_best_b_op_s41_3090
```

## 输出

- 机器: lab3090 (3090 24G, docker 18fbbab202e1, pwrlim 280W 保持)
- 预计时长: ~12-14h (Base OP + WITH_CP on 3090, 参考 exp263d 14h50min Base OD)
- ETA: 14:29 → 后天 04:29 CST FINAL

## 对照

| Exp | seed | 机器 | FINAL mAP/R1 |
|-----|------|------|--------------|
| exp264 | 42 | srvC Tiny | 76.7 / 85.1 |
| exp265 | 42 | srvC Small | 78.4 / 86.2 |
| exp266 | 42 | srvC Base | e60 eff 78.4/86.2 (silent exit) |
| **exp265b** | 41 | srvA Small | pending (tmr 06:10) |
| **exp266b (srvA)** | 41 | srvA Base | pending (after exp265b) |
| **exp266b_3090 (本)** | **41** | **lab3090 Base** | pending (tmr ~04:29) |

## 预期

- 若 exp266b_3090 FINAL > 78.5/86.5: Base OP 终于显示对 Small 的增益
- 和 exp265b/exp266b 三方对照 (Small vs Base, srvA vs lab3090 设备差)
- 修 exp266 主表 "e60 eff" 瑕疵, 给 e120 完整 FINAL
