# exp269 monitor — Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD (PLBOA OFF) @ Market-1501

**第 2 个 Base run** (原计划 local 3090,3090 挂,改 5060 Ti with_cp)

- 机器: srvA (i-2.gpushare.com:29162, 5060 Ti 16G)
- 启动: 2026-04-20 00:40:16 (auto-chained by queue_on_ckpt.sh daemon 3901347 from exp268 → exp269,新 main PID 4170236)
- Log: /hy-tmp/log/market/exp269_best_b_m_s42/train_log.txt
- Config: configs/market/prcv_best_base.yml (WITH_CP=True, PLBOA OFF)

## 对照

- 旧协议 exp260b Base Market = 94.4 / 97.1 (本地 3090,无默认 flip-test)
- 新协议加 default flip-test 期望 +0.2~0.5,目标 ≥94.6
- 同期 Small exp268 (刚完成) = 94.3 / 97.3 → Base 应该 ≥ Small,目标 ≥94.5

## 中间 eval

| Epoch | mAP | R-1 | 备注 |
|-------|-----|-----|------|
| e1 | 冷启动 | loss 14.7 acc 0 | warmup |

## 自动化状态

- srvA 后续无 daemon 排队 (L0 queue_next + L1 queue_on_ckpt 均已完成使命)
- 下一个 chain 要人工补: Phase 3-A exp274/275/276/277 Small PSG stage 消融

## 预期 ETA

- Base with_cp ~10.7min/epoch,120 epoch ≈ 21h
- 预计 2026-04-20 ~22:00 完成 → srvA 空闲,届时起 Phase 3-A Small 4 runs
