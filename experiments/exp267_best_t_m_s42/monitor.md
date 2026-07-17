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

## FINAL (e120) — 2026-04-19 13:45:45 srvB

- **mAP: 92.5%**
- **CMC Rank-1: 96.4%**
- CMC Rank-5: 98.9%
- CMC Rank-10: 99.3%
- ckpt: `/hy-tmp/log/market/exp267_best_t_m_s42/transformer_120.pth`

### 轨迹

| Epoch | mAP |
|-------|-----|
| 10 | 83.1 |
| 20 | 87.3 |
| 30 | 89.5 |
| 40 | 90.6 |
| 50 | 91.4 |
| 60 | 91.7 |
| 70 | 92.0 |
| 80 | 92.3 |
| 90 | 92.3 |
| 100 | 92.5 |
| 110 | 92.5 |
| **120** | **92.5** |

最后 3 次 eval 全 92.5,完全收敛。

### 对照与解读

- **对照历史 `4090-M-PSG-small-lr8 = 93.7`**(那是 Small,不是 Tiny)→ Tiny + full scaffold 92.5 略低于 Small PSG-only,合理(Tiny 容量小)
- 对照 `4090-M-base` baseline 91.6 → 我们 Tiny full scaffold +0.9
- Tiny Market 在 full scaffold 上触 92.5,足够支撑论文主表 Tiny 行

### 结论

- Swin-Tiny + 2-stage PSG + LGPA-D + GCN512 + OA-SD (PLBOA OFF,Market 非 occluded) + default flip-test @ Market-1501 = **92.5 / 96.4**
- 待 exp268/exp269 出 Small/Base 数后三 backbone Market 比较一起写

### 后续

- queue_on_ckpt.sh daemon (srvB PID 26925) 检测到 transformer_120.pth + 确认进程退出,于 13:46:11 自动起 **exp263 Base OD (PID 34335)** — 第一个 Base run!
- srvB 后续 chain: exp263(17h) → 空闲 (→ Phase 3-A srvB 拿 Tiny 4 runs)
