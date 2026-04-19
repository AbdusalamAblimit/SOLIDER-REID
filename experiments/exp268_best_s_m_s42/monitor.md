# exp268 monitor — Swin-Small + 2-stage PSG + LGPA-D + GCN512 + OA-SD (PLBOA OFF) @ Market-1501

- 机器: srvA (i-2.gpushare.com:29162, 5060 Ti 16G)
- 启动: 2026-04-19 10:00:48 (auto-chained by queue_next.sh daemon 3874825 from exp262 → exp268，新 main PID 3968073)
- Log: /hy-tmp/log/market/exp268_best_s_m_s42/train_log.txt
- Config: configs/market/prcv_best_small.yml
- PLBOA: OFF（Market 非 occluded，per phase1_design.md）

## 对照

- 4090 历史 Swin-Small + PSG on Market LR4 = 93.9/96.9, LR8 = 93.7/96.9 (baseline PSG only，不带 full scaffold)
- exp260b Base Market FINAL = 94.4/97.1 (full scaffold 旧协议)
- exp267 Tiny Market 当前 e70 = 92.0 (ongoing)
- 目标 exp268 Small Market ≥94/97,**超过 exp260b Base 是目标**（Small 新协议 + default flip-test 期望 +0.3-0.5）

## 中间 eval

| Epoch | mAP | R-1 | 备注 |
|-------|-----|-----|------|
| e3 | 冷启动 | loss 11.7 acc 0.03 | warmup 正常 |

## 自动化状态

- queue_next.sh daemon (srvA PID 3874825) 完成使命,已退
- queue_on_ckpt.sh daemon (srvA PID 3901347) 继续等 exp268 的 transformer_120.pth,触发后起 exp269 (Base Market)
- Monitor b1ksod4yh 持续 tail 三台日志

## 预期 ETA

- Small 在 5060 Ti 上约 8min/epoch(旧 exp262 = 8min),120 epoch ≈ 16h(比 Tiny Market 慢 ~2x)
- 预计 2026-04-20 02:00 前后完成 → 随后自动启 exp269
