# exp155 Evidential DL 监控

## 实验信息
- 方法: Evidential Deep Learning (Dirichlet 分类头) for GCN branch
- 基线: exp030a-eq (60.73% mAP 3-seed mean)
- 运行位置: 本地 3090
- CHECKPOINT_PERIOD: 20

## 监控记录

### [2026-03-23 08:39] ep10 评估 — 正向

| Epoch | exp155 (Evidential) | exp030a | Δ |
|-------|--------------------|---------|----|
| 10 | 38.9 / 52.9 | 38.2 / 51.3 | **+0.7 / +1.6** |

- `evid_br = 6.15`（与 CE 量级一致）
- `evid_unc = 0.577`（未变化——evidence 收敛极慢）
- `evid_kl = 63.7, evid_ann = 0.153`（KL 退火刚开始）
- `id_part = 7.12` > 标准 CE 的 ~6.7，Evidential loss 值稍高
- 下一关键点：ep20/30

### [2026-03-23 09:00] ep20/30 — 正向在收窄

| Epoch | exp155 | exp030a | Δ mAP | Δ R1 |
|-------|--------|---------|-------|------|
| 10 | 38.9 / 52.9 | 38.2 / 51.3 | +0.7 | +1.6 |
| 20 | 47.1 / 59.8 | 46.8 / 60.9 | +0.3 | -1.1 |
| 30 | 52.4 / 64.6 | 52.2 / 66.0 | +0.2 | -1.4 |

- `evid_unc = 0.581`（从 0.577 几乎没变——evidence 收敛极慢）
- `id_part = 8.0+`（远高于 CE 同期 ~2.0，说明 Evidential 梯度太弱）
- 正向从 +0.7 收窄到 +0.2，趋势不乐观
- R1 持续负向（-1.4）
- 根本问题可能是 Evidential Bayes Risk 的梯度量级 < CE，导致 GCN branch 学习速度变慢
