# exp266 monitor — Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA @ Occluded-PoseTrack-ReID

**第 3 (最后 1) 个 Base run** (原计划 local 3090,3090 挂,改 5060 Ti with_cp)

- 机器: srvC (i-2.gpushare.com:25551, 5060 Ti 16G)
- 启动: 2026-04-20 04:46:14 (auto-chained by queue_on_ckpt.sh daemon 34381 from exp265 → exp266,新 main PID 49593)
- Log: /hy-tmp/log/occluded_posetrack/exp266_best_b_op_s42/train_log.txt
- Config: configs/occluded_posetrack/prcv_best_base.yml (WITH_CP=True, PLBOA ON)

## 对照

- 同一 scaffold: exp264 Tiny OP = 76.7/85.1, exp265 Small OP = 78.4/86.2
- 期望 Base > Small,目标 ≥79/87 on Occ-PTrack
- KPR 在 Occ-PTrack 的 baseline 数字需从 KPR paper Table 补上用于对比

## 中间 eval

| Epoch | mAP | R-1 | 备注 |
|-------|-----|-----|------|
| e1 | 冷启动 | loss 11.7 acc 0.002 | warmup |

## 自动化状态

- srvC 后续无 daemon 排队(L0 + L1 + L2 都已完成使命并退出)
- 下一个 chain 要人工补: Phase 3-B Tiny/Small GCN 消融

## 预期 ETA

- Base OP: Small OP 单 epoch 628s → Base OP 应 ~670s/epoch (with_cp 额外开销)
- 120 epoch ≈ 22h
- 预计 2026-04-21 ~02:46 完成 → srvC 空闲后起 Phase 3-B

## Phase 1 + Base 进度

完成 Phase 1 Tiny/Small 6 run + 2 个 Base run(exp263/269 进行中),exp266 是最后 1 个 Base。全部完成后 Phase 1 九格填满(Base 三行用新协议数字)。
