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

## FINAL (e120) — 2026-04-20 00:39:02 srvA

- **mAP: 94.3%**
- **CMC Rank-1: 97.3%**
- CMC Rank-5: 99.1%
- CMC Rank-10: 99.5%
- ckpt: `/hy-tmp/log/market/exp268_best_s_m_s42/transformer_120.pth`

### 轨迹

| Epoch | mAP | R-1 |
|-------|-----|-----|
| 10 | 89.4 | 95.0 |
| 20 | 91.5 | 96.0 |
| 30 | 92.5 | 96.5 |
| 40 | 93.3 | 96.9 |
| 50 | 93.3 | 96.6 |
| 60 | 93.8 | 97.2 |
| 70 | 94.1 | 97.1 |
| 80 | 94.1 | 97.1 |
| 90 | 94.2 | 97.1 |
| 100 | 94.3 | 97.3 |
| 110 | 94.3 | 97.2 |
| **120** | **94.3** | **97.3** |

最后 3 次 eval (e100/110/120) mAP 全 94.3,R1 97.3/97.2/97.3。已收敛到平台。

### 对照

- **exp267 Tiny Market FINAL = 92.5 / 96.4** → Small +1.8 mAP / +0.9 R1
- **exp260b Base Market FINAL (旧协议) = 94.4 / 97.1** → Small 新协议 94.3 / 97.3,mAP 差 0.1,但 R1 超 +0.2(新协议 flip-test 作用)
- 对照 4090-M-PSG-Small-lr4 = 93.9/96.9(仅 PSG,无 full scaffold)→ 新协议 full scaffold +0.4 mAP / +0.4 R1
- 基本和 Base 旧协议接近 — 新协议 Small 已基本逼近旧协议 Base 的 Market 水平

### 结论

- Swin-Small + 2-stage PSG + LGPA-D + GCN512 + OA-SD (PLBOA OFF) + default flip-test @ Market-1501 = **94.3 / 97.3**

### 后续

- queue_on_ckpt.sh daemon (srvA PID 3901347) 检测 transformer_120.pth + 无 crash,于 00:40:16 自动起 **exp269 Base Market (PID 4170236)** — 第 2 个 Base run!
- srvA 后续: exp269(17h) → 空闲 (→ Phase 3-A Small 或 3-B)
