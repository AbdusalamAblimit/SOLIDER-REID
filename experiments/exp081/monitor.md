# exp081 PQTD 训练监控

## 实验信息
- **方法**: PSG + PAA + PQTD (Pose-Query Transformer Decoder, 替代 GCN)
- **配置**: `configs/occluded_duke/pose_psg_paa_pqtd.yml`
- **对照**: exp066 PAA (PSG+GCN+PAA) = 61.6%/74.2%
- **核心改动**: 用 3-layer transformer decoder + 5 pose-guided queries 替代 2-layer GCN
- **参数**: ~2.5M (vs GCN ~400K)
- **启动**: 2026-03-16 17:01, PID 3378400, 本地 3090

---

### [17:12] Ep10 首次评估 ⚠️
- **mAP 31.8% / R1 41.9%** — vs PAA ep10 38.4%/51.8%, **Δ -6.6%/-9.9%**
- id_part 5.98 (vs GCN ~4.5), id_global 5.67 (vs GCN ~3.5) — 收敛显著慢
- Decoder 太重导致优化困难？还是需要更长 warmup？
- 不提前终止，观察到 Ep30-40 再判断

| Epoch | PQTD mAP | PQTD R1 | PAA ref mAP | PAA ref R1 |
|-------|----------|---------|-------------|-----------|
| 10 | 31.8% | 41.9% | 38.4% | 51.8% |
| 20 | 40.6% | 50.4% | ~47.5% | ~60.6% |

| 30 | 46.4% | 56.6% | ~52.1% | ~64.5% |

| 40 | 50.2% | 59.5% | ~56.3% | ~68.6% |

### [17:44] Ep40 — 追赶继续减速
- 增量: +8.8→+5.8→+3.8 mAP per 10 epoch，递减
- 仍落后 PAA ~6%/~9%
- 按趋势外推: 最终 ~58-59%（低于 PAA 61.6%）
- 但 decoder warmup 可能在 Ep50+ 有 phase change
- id_part 1.07，接近 GCN 的最终水平

---
