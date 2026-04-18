# exp264 monitor — Swin-Tiny + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA @ Occluded-PoseTrack

- 机器: srvC (i-2.gpushare.com:25551, 5060 Ti 16G)
- 启动: 2026-04-18 ~17:30 (main PID 25163, etime 9h39m at first check)
- Log: /hy-tmp/log/occluded_posetrack/exp264_best_t_op_s42/train_log.txt
- Config: configs/occluded_posetrack/prcv_best_small.yml → 注: 实际 config 是 tiny，路径可能有笔误需再确认

> 修正确认: process cmd 显示 `--config_file configs/occluded_posetrack/prcv_best_tiny.yml`。正确。

- 前序历史: 带 LGPA NaN fix + srvC 特有的 `datasets/occluded_posetrack.py` fix（也已进 commit e6150e5）

## 中间 eval（每 10 epoch，带 flip-test）

| Epoch | mAP | R-1 | R-10 | 备注 |
|-------|-----|-----|------|------|
| 60 | 76.3% | 84.7% | 96.8% | Occ-PTrack 数据集特性导致数字比 OD 高 |
| 70 | 76.5% | 84.8% | 97.0% | 稳定小步上升 |
| 80 | 76.7% | 85.2% | 96.9% | ↑ |

预期 final (e120) 76-77/85-86。对标 KPR w/o prompt 在 Occ-PTrack 上的数字（需从 KPR Table 补录）。

## Loss 健康性（e83 处抽样）

- Total Loss 3.17, Acc 0.992
- id_global 0.147, id_part 2.07, tri_global 0.003, tri_part 0.473, oa_sd 0.043, lgpa_assign 3.513
- 全部正常。GPU mem 5.7GB/16GB，util 91%。

## 自动化状态

- Monitor b1ksod4yh tail 中
- queue_next.sh daemon (srvC PID 32839) wait main PID 25163 → 下一个 exp265 (Small OccPTrack)
- 30min 心跳 cron 覆盖

## 状态决策

- 2026-04-19 03:00: 正常，ETA ~4.5h 到 epoch 120。
- Occ-PTrack 是 secondary benchmark，目标只要不明显低于 KPR w/o prompt 就 OK；当前数字看上去很稳。
