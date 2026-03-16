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

### [17:33] Ep30 — 追赶放缓
- Ep10→20→30: +8.8→+5.8 mAP，追赶减速
- 仍落后 PAA ~6%
- 但 decoder 的 id_part 从 5.98→2.05 快速收敛
- 继续到 Ep60 看是否能追平

---
