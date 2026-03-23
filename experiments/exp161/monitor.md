# exp161 STD-PR 监控

## 实验信息
- 方法: Structural Token Decomposition with Pose-guided Routing
- 类型: 范式级创新（spatial tokens → structural body-part tokens）
- 基线: exp030a-eq (60.73% mAP 3-seed mean)
- 运行: 本地 3090（等 seed42 完成后启动）
- CHECKPOINT_PERIOD: 20

## 核心关注指标
- equal_concat mAP/R1（与 GCN branch 直接对比）
- token_norm（structural tokens 的范数，监控是否有 collapse）

## 与历史 decoder 实验的关键区别
- exp063 PTD: 2 层 decoder, dim=256, 120ep 不够 → mAP 56.7%
- exp081 PQTD: 3 层 decoder, dim=256, 120ep 不够 → mAP 56.9%
- **exp161 STD-PR**: 2 层 cross-attn, **dim=768**(不降维), **pose heatmap additive bias**
- 如果 dim=768 + pose bias 能让 cross-attn 收敛更快 → 有可能在 120ep 内有效

## 监控

### [2026-03-23 14:36] ep10 — 低于 baseline，符合 decoder 模式
- mAP 35.2% / R1 44.5%
- vs exp030a ep10: -3.0 / -6.8
- id_part = 4.22（从 6.73 下降，比 GCN 同期 ~5.5 更低 → 学得更快）
- tri_part = 0.50（远低于 GCN ~2.3，structural tokens 特征更紧凑）
- 模式与 exp063 PTD / exp081 PQTD 一致（decoder 早期慢）
- **关键**：ep30-50 能否追回到 baseline

### [2026-03-23 14:57] ep20/30 — 追赶中，远超历史 decoder

| Epoch | STD-PR | exp030a | Δ mAP | exp063 PTD 同期 |
|-------|--------|---------|-------|----------------|
| 10 | 35.2 | 38.2 | -3.0 | ~25% |
| 20 | 44.3 | 46.8 | -2.5 | ~33% |
| 30 | 50.2 | 52.2 | -2.0 | ~37% |

- 与 exp063 PTD 相比：STD-PR ep30=50.2% vs PTD ep30≈37%（+13%！）
- dim=768 + pose heatmap bias 确实大幅加速 cross-attention 收敛
- 但 R1 仍然落后 6.3 个点
- 如果 ep60-80 能追到 -0.5 以内 → STD-PR 有望超越 GCN
