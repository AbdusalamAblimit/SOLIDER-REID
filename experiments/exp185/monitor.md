# exp185 SupCon on STD-PR pooled 监控

## 检查点

### [10:20] 检查点 #1
### [19:42] 检查点 #3
### [20:46] ep90: 61.4/72.4. vs exp166 (CE per-token) 62.1/74.0 = -0.7/-1.6.
## 最终结果

| Epoch | mAP | R1 |
|-------|------|------|
| 80 | 61.5% | 71.8% |
| 90 | 61.4% | 72.4% |
| 100 | 62.7% | 73.5% |
| 110 | 62.9% | 73.4% |
| **120** | **62.8%** | **73.3%** |

vs per-token+SupCon (exp176): 64.1/75.5 → **per-token: +1.3/+2.2**
vs per-token+CE (exp166): 63.1/73.9 → pooled+SupCon: **-0.3/-0.6**

**Pooled SupCon 不如 per-token CE！SupCon 需要 per-token structure。**
