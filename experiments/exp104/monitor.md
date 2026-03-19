# exp104 PACD 监控

## 基线: exp066 61.6%/74.2%

### v1 (bug 版): mask 76%，loss 50+，R1 +1.6% (artifact of extreme dropout)
### v3 (修正版): 3×3 mask, cosine loss

| Ep | mAP | R1 | pacd loss | 备注 |
|----|------|------|-----------|------|
| 10 | 38.4% | 51.1% | — | warmup 期间 |
| 11+ | — | — | 0.022 | 极小，几乎无学习信号 |

PACD loss 太小 (~0.02)：12×4 特征图 GAP 对 30% mask 鲁棒。
v3 大概率中性（PACD 无实际效果）。
