# exp263 monitor — Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA @ Occluded-Duke

**第一个 Base run** (原计划 local 3090，3090 挂了改走 5060 Ti with_cp)

- 机器: srvB (i-1.gpushare.com:61604, 5060 Ti 16G)
- 启动: 2026-04-19 13:46:11 (auto-chained by queue_on_ckpt.sh daemon 26925 from exp267 → exp263，新 main PID 34335)
- Log: /hy-tmp/log/occluded_duke/exp263_best_b_od_s42/train_log.txt
- Config: configs/occluded_duke/prcv_best_base.yml (WITH_CP=True)

## Base backbone 特殊说明

- `MODEL.WITH_CP: True` — gradient checkpointing,显存只 6-8GB
- 2026-04-19 13:47 首次 iter 显示 GPU mem 10.1GB (含 eval buffers),稳定后应回落到 7-8GB
- 预计单 epoch ~8.5min (Tiny 4.5min,Small 8min,Base 8.5min with_cp),120 epoch ≈ 17h
- 参数量 88M (vs Tiny 28M, Small 50M)

## 对照

- 旧协议 exp260b Base @ OD (本地 3090) = 73.9/83.2
- 新协议加 default flip-test 期望 +0.5~0.9,目标 ≥74.4/83.5
- 对标 KPR w/o prompt (Swin-B) = 73.3/82.5 → 期望**明显超过**

## 中间 eval

| Epoch | mAP | R-1 | 备注 |
|-------|-----|-----|------|
| e1 | 冷启动 | loss 20 acc 0.001 | warmup; tri_global=19 极高反映 Base 未收敛的 metric learning |

## 自动化状态

- srvB 当前无 daemon 在队列后面 — 之前的 queue daemon(L0/L1/L2)都已完成使命退出
- 下一个 chain 要人工补: Phase 3-A exp270/271/272/273 Tiny PSG stage 消融
- Monitor b1ksod4yh 持续 tail

## 预期 ETA

- 2026-04-20 ~07:00 前后完成 → srvB 空闲,届时起 Phase 3-A Tiny 4 runs
