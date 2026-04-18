# exp265 monitor — Swin-Small + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA @ Occluded-PoseTrack-ReID

- 机器: srvC (i-2.gpushare.com:25551, 5060 Ti 16G)
- 启动: 2026-04-19 07:16:46 (auto-chained by queue_next.sh daemon PID 32839 from exp264 → exp265，新 main PID 36015)
- Log: /hy-tmp/log/occluded_posetrack/exp265_best_s_op_s42/train_log.txt
- Config: configs/occluded_posetrack/prcv_best_small.yml

## 对照

- exp264 Tiny @ Occ-PTrack FINAL = 76.7/85.1 (刚出)
- 目标: Small 应超过 Tiny；对照 KPR Table 的 Small 行
- 期望 final: ≥78/86

## 中间 eval

| Epoch | mAP | R-1 | 备注 |
|-------|-----|-----|------|
| e1 | 冷启动 | loss 13 acc 0 | 启动正常 |

## 自动化状态

- queue_next.sh daemon (srvC PID 32839) 完成使命(exp264→exp265 单次),已退出
- queue_on_ckpt.sh daemon (srvC PID 34381) 继续等 exp265 的 transformer_120.pth 出现，触发后自动起 exp266 (Base OP)
- Monitor b1ksod4yh 继续 tail，cron 52ba1096 每 30min 简报
