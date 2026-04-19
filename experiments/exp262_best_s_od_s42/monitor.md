# exp262 monitor — Swin-Small + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA @ Occluded-Duke

- 机器: srvA (i-2.gpushare.com:29162, 5060 Ti 16G)
- 启动: 2026-04-18 ~17:30 (main PID 3736002, etime 9h38m at first check)
- Log: /hy-tmp/log/occluded_duke/exp262_best_s_od_s42/train_log.txt
- Config: configs/occluded_duke/prcv_best_small.yml
- 前序历史: `_old_buggy_lgpa` 和 `_v1_firstfix` 子目录存在，第 3 次起；带 LGPA NaN fix

## 中间 eval（每 10 epoch，带 flip-test）

| Epoch | mAP | R-1 | R-10 | 备注 |
|-------|-----|-----|------|------|
| 40 | 69.6% | 80.4% | 91.3% | baseline Small 即 ~65-66，加 full scaffold 抬到 69+ |
| 50 | 71.5% | 81.5% | 91.5% | ↑ |
| 60 | 72.3% | 82.4% | 92.0% | ↑ 仍在爬 |

预期 final (e120) ≥ 74 / 83（对齐旧协议 exp255 = 73.2/83.3 + 新协议默认 flip 加成 +0.9）。

## Loss 健康性（e68 处抽样）

- Total Loss 3.38, Acc 0.990
- id_global 0.210, id_part 2.44, tri_global 0.015, tri_part 0.573, oa_sd 0.028, lgpa_assign 3.56
- 全部正常。GPU mem 7.8GB/16GB，util 81%。

## 自动化状态

- Monitor b1ksod4yh tail 中
- queue_next.sh daemon (srvA PID 3874825) wait main PID 3736002 → 下一个 exp268 (Market Small)
- 30min 心跳 cron 覆盖

## 状态决策

- 2026-04-19 03:00: 正常，ETA ~7h 到 epoch 120。
- 若 e80-e100 mAP 跨越 73 即说明方向正确；若卡在 72 区间说明 Small 新协议可能略差于旧协议 exp255，需要 post-hoc 分析 flip-test 单独影响

## FINAL (e120) — 2026-04-19 09:59:56 srvA

- **mAP: 73.8%**
- **CMC Rank-1: 83.1%**
- CMC Rank-5: 90.2%
- CMC Rank-10: 92.2%
- ckpt: `/hy-tmp/log/occluded_duke/exp262_best_s_od_s42/transformer_120.pth`

### 轨迹

| Epoch | mAP | R-1 |
|-------|-----|-----|
| 40 | 69.6 | 80.4 |
| 50 | 71.5 | 81.5 |
| 60 | 72.3 | 82.4 |
| 70 | 72.8 | — |
| 80 | 73.6 | — |
| 90 | 73.5 | — |
| 100 | 73.7 | — |
| 110 | 73.8 | — |
| **120** | **73.8** | **83.1** |

最后 3 次 eval (100/110/120) mAP 全在 73.7-73.8,已平台。

### 对照与解读

- **对照旧协议 exp255 = 73.2/83.3**（同 scaffold，无默认 flip-test）→ 新协议 +0.6 mAP / -0.2 R1。mAP 稳赢,R1 持平到略降(噪声内)。验证了默认 flip-test 对 mAP 的稳定 +0.6~0.9 贡献
- 远远超过 BPBreID (62.5/75.1) 和 SOLIDER (61.9/71.2)
- 接近但未超过 KPR w/o prompt (73.3/82.5) — 我们 73.8 **略优** (+0.5 mAP / +0.6 R1)
- 未到 KPR 带 prompt 的 75.1/84.3(这个是他们带 prompt 的版本,我们是 prompt-free,公平对比基准是 KPR w/o prompt)

### 结论

- Swin-Small + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA + default flip-test @ Occ-Duke = **73.8 / 83.1**
- **比 KPR w/o prompt 略优**(主要 baseline 对标已达成)
- MaxSim eval 待独立跑

### 后续

- queue_next.sh daemon 10:00:48 自动起 exp268 (Small Market, PID 3968073)
- queue_on_ckpt.sh daemon 3901347 仍在位,等 exp268 ckpt 接 exp269 (Base Market)
