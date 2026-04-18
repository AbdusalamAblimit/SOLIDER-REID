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

## FINAL (e120) — 2026-04-19 04:16:04 srvB

- **mAP: 65.9%**
- **CMC Rank-1: 77.4%**
- CMC Rank-5: 86.9%
- CMC Rank-10: 89.5%
- ckpt: `/hy-tmp/log/occluded_duke/exp261_best_t_od_s42/transformer_120.pth`

相对期望(≥60/72)超出 +5.9/+5.4。相对旧协议同 scaffold (exp255 Small = 73.2/83.3 @ Swin-Small) 不可直接比，这是 Tiny。对比历史 Tiny 最佳：
- 4090 Tiny `4090-OD-PSG-small-lr8` baseline = 65.8 (但那是 Small lr8 base)
- 本项目 Tiny 实际对应 `exp000 baseline` 56.6 + 各种 scaffold；exp261 **65.9 是 Tiny + full scaffold 的新高**。

### 结论

- Tiny + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA + default flip-test @ OccDuke = **65.9 / 77.4**
- 最后 ~20 epoch 在 65.8-65.9 平稳，无明显过拟合
- 不做 MaxSim eval 不写入 PRCV 主表的 MaxSim 行；等 queue daemon 将 srvB 排到后续实验后，找时间回头用 test.py 跑一次 MaxSim

### 后续

- queue_next.sh daemon 已识别 transformer_120.pth 存在 + 无 crash，于 04:16:43 自动起 exp267 (Market Tiny，PID 25860)
- srvB 接下来 run exp267，不要回跑 exp261 any more
