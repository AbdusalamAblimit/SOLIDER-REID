# exp166 STD-PR Per-Token + PLBOA

- 6 tokens each independently classified
- test: 6 tokens L2-norm concatenated (global_768 + 6×768 = 5376-d)

## Bug 已修复: tri_part=inf → L2 normalize per-token features

## 训练监控结果

| Epoch | mAP | R1 | R5 | R10 |
|-------|------|------|------|------|
| 80 | 61.5% | 73.2% | 84.9% | 88.6% |
| 90 | 62.1% | 74.0% | 85.6% | 88.9% |
| 100 | 62.9% | **74.5%** | 86.2% | 89.3% |
| 110 | 63.0% | 74.0% | 86.0% | 89.3% |
| 120 | **63.1%** | 73.9% | 86.1% | 89.5% |

## 最终结论

- **mAP 63.1%** — 当前所有单 seed 实验中最高
- **R1 73.9%** — 与 PLBOA+GCN 3-seed 均值持平
- 峰值 R1 74.5%@ep100，后续略降 → 收敛正常
- vs STD-PR+PLBOA V1 mean-pool (3-seed 62.6/72.7): **mAP +0.5, R1 +1.2**（单 seed vs 3-seed mean）
- vs PLBOA+GCN (3-seed 62.1/73.9): **mAP +1.0, R1 ±0**

Per-token classification 是 STD-PR 目前最佳配置。
