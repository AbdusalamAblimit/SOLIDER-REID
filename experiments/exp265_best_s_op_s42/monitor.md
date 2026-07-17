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

## FINAL (e120) — 2026-04-20 04:45:45 srvC

- **mAP: 78.4%**
- **CMC Rank-1: 86.2%**
- CMC Rank-5: 94.8%
- CMC Rank-10: 97.3%
- ckpt: `/hy-tmp/log/occluded_posetrack/exp265_best_s_op_s42/transformer_120.pth`

### 轨迹

| Epoch | mAP | R-1 |
|-------|-----|-----|
| 10 | 74.6 | 82.9 |
| 20 | 75.9 | 83.9 |
| 30 | 77.0 | 84.5 |
| 40 | 77.8 | 85.7 |
| 50 | 78.0 | 85.5 |
| 60 | 78.3 | 85.8 |
| 70 | 78.3 | 85.8 |
| 80 | 78.3 | 85.9 |
| 90 | 78.3 | 85.9 |
| 100 | 78.4 | 86.1 |
| 110 | 78.4 | 86.2 |
| **120** | **78.4** | **86.2** |

最后 5 次 eval (e80-e120) mAP 平台 78.3-78.4,R1 85.9-86.2,已完全饱和。

### 对照

- **exp264 Tiny OP FINAL = 76.7 / 85.1** → Small +1.7 mAP / +1.1 R1
- Occ-PTrack 作为 secondary benchmark,Small 78.4/86.2 是当前项目 Occ-PTrack 最强单 seed

### 结论

- Swin-Small + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA + default flip-test @ Occ-PTrack-ReID = **78.4 / 86.2**

### 后续

- queue_on_ckpt.sh daemon (srvC PID 34381) 于 04:46:14 自动起 **exp266 Base OP (PID 49593)** — 第 3 (最后 1) 个 Base!
- srvC 后续: exp266(~22h)→ 空闲 → Phase 3-B
