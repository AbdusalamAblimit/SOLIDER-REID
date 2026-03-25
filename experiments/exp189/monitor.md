# exp189 Visibility-Weighted SupCon 监控

## 检查点

### [01:03] 检查点 #1
### [01:19] 检查点 #2
### [01:37] 检查点 #3
### [01:55] 检查点 #4

| Epoch | vis-weighted | uniform (exp176) | delta |
|-------|------|------|-------|
| 20 | 48.5/60.4 | 46.7/60.5 | +1.8/-0.1 |
| 30 | 52.9/65.1 | 52.9/63.8 | ±0/+1.3 |

## 最终结果

| Epoch | mAP | R1 |
|-------|------|------|
| 80 | — | — |
| 90 | — | — |
| 100 | — | — |
| 110 | — | — |
| **120** | **63.3%** | **73.7%** |

vs exp176 (uniform SupCon): 64.1/75.5 → **-0.8/-1.8**

**结论**: Visibility-weighted SupCon 是负向的。Uniform weighting 就是最优。
Per-part visibility weighting 反而伤害了 R1（-1.8）。可能是因为降低遮挡区域的 contrastive 信号
恰恰减少了模型在 PLBOA 环境下学习的机会 — 与 SupCon × PLBOA synergy 的核心机制矛盾。
