# exp261 monitor — Swin-Tiny + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA @ Occluded-Duke

- 机器: srvB (i-1.gpushare.com:61604, 5060 Ti 16G)
- 启动: 2026-04-18 ~17:30 (main PID 16336, etime 9h40m at first check)
- Log: /hy-tmp/log/occluded_duke/exp261_best_t_od_s42/train_log.txt
- Config: configs/occluded_duke/prcv_best_tiny.yml
- 前序历史: 同目录下有 `_old_buggy_lgpa` 和 `_v1_firstfix` 两个目录，说明此轮是第 3 次起，前两次因 LGPA NaN 提前失败；当前 run 带 attn score clamp + log-softmax 数值稳定 fix（已进 commit e6150e5）

## 中间 eval（每 10 epoch，带 flip-test）

| Epoch | mAP | R-1 | R-10 | 备注 |
|-------|-----|-----|------|------|
| 80 | 64.9% | 76.3% | 89.5% | 稳步上升 |
| 90 | 65.9% | 77.6% | 89.8% | ↑ |
| 100 | 65.8% | 77.2% | 89.8% | 微降 0.1 mAP / 0.4 R1，LR 降到 3.2e-5 cosine 尾 |

预期 final (e120) ≥ 65.5 / 77，符合 design.md 的 ≥60/72 目标。

## Loss 健康性（e105 处抽样）

- Total Loss 3.13, Acc 0.991
- id_global 0.183, id_part 2.22, tri_global 0.019, tri_part 0.577, oa_sd 0.040, lgpa_assign 3.116
- 全部在正常区间，无 NaN / Inf

## 自动化状态

- Monitor b1ksod4yh 持续 tail，过滤 Traceback/OOM/Killed/NaN/Inf/FINAL
- queue_next.sh daemon (srvB PID 25015) 已挂起，wait main PID 16336；训练结束且 transformer_120.pth 存在且无 crash → 自动起 exp267 (Market Tiny)
- 30min 心跳 cron ac534e44 每 13/43 分钟自检

## 状态决策

- 2026-04-19 03:00: 正常，无需干预。ETA 约 1.5h 到 epoch 120。
