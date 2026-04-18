# exp267 monitor — Swin-Tiny + 2-stage PSG + LGPA-D + GCN512 + OA-SD (PLBOA OFF) @ Market-1501

- 机器: srvB (i-1.gpushare.com:61604, 5060 Ti 16G)
- 启动: 2026-04-19 04:16:43 (auto-chained by queue_next.sh daemon PID 25015 from exp261 → exp267，新 main PID 25860)
- Log: /hy-tmp/log/market/exp267_best_t_m_s42/train_log.txt
- Config: configs/market/prcv_best_tiny.yml
- PLBOA: OFF（Market 非 occluded 数据集，per phase1_design.md）

## 中间 eval

| Epoch | mAP | R-1 | 备注 |
|-------|-----|-----|------|
| e0-2 | 冷启动 | — | loss 12-13，acc ~0.3%；early warmup |

## 自动化状态

- queue_next.sh daemon 25015 已完成使命（exp261→exp267 转换成功）
- Monitor b1ksod4yh 继续 tail 三台 log
- 新 cron 52ba1096 动态检测当前 exp，无需手动调 exp 名

## 预期

- Market 上 Tiny + scaffold 期望 ≥90/95（对照 4090-M-PSG-small-lr4 = 93.9，Tiny 会略低）。

## 下一步

- 训练完 120 epoch 后（~8h on 5060 Ti）无下一个 run，srvB 空闲
- 空闲后如果 Phase 1 全盘完成，srvB 可进 Phase 3-A exp270/271/272/273（Tiny PSG stage 消融）
- 或等 Base 3 run 被批准后接一个 Base
