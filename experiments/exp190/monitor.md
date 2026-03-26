# exp190 3-view Parallel Aug + CE 监控

配置: triple injection + PLBOA + parallel aug + CE (无 SupCon)
对照: exp166 (1-view + CE): 63.1/73.9 | exp187 (3-view + SupCon): 64.9/76.6

## 检查点

### [23:31] 检查点 #1

**状态**: 正常
**进度**: Epoch 1/120

| 指标 | 当前值 |
|------|--------|
| ID Global | 6.554 |
| ID Part | 6.703 |
| Tri Global | 13.96 |
| Speed | ~100 samples/s (3-view) |

**观察**: 3-view + CE 正常启动。ETA ~4.5h。
### [23:55] 检查点 #2

**状态**: 正常
**进度**: Epoch 10/120

| 实验 | ep10 mAP | ep10 R1 |
|------|----------|---------|
| exp190 (3-view+CE) | **38.9%** | **51.8%** |
| exp187 (3-view+SupCon) | 38.3% | 52.0% |
| exp176 (1-view+SupCon) | 34.6% | 48.0% |
| exp166 (1-view+CE) baseline | ~36% | ~48% |

**观察**: 3-view+CE ≈ 3-view+SupCon at ep10。3-view 的加速效果与 loss type 无关。
### [00:20] 检查点 #3

**状态**: 正常
**进度**: Epoch 20/120

| 实验 | ep10 | ep20 |
|------|------|------|
| exp190 (3-view+CE) | 38.9/51.8 | **49.1/60.7** |
| exp187 (3-view+SupCon) | 38.3/52.0 | 49.8/61.8 |

**观察**: CE 开始落后 SupCon (-0.7/-1.1)。SupCon 在 3-view 下仍有优势。
Remote exp191 ep90: 62.4/75.1 (OA-SD+CE 超过 exp166!)
### [00:44] 检查点 #4

**状态**: 正常
**进度**: Epoch 30/120

| Epoch | 3-view+CE (exp190) | 3-view+SupCon (exp187) | delta |
|-------|------|------|-------|
| 10 | 38.9/51.8 | 38.3/52.0 | +0.6/-0.2 |
| 20 | 49.1/60.7 | 49.8/61.8 | -0.7/-1.1 |
| 30 | 56.7/68.5 | 57.7/70.1 | -1.0/-1.6 |

**观察**: CE 在 3-view 下持续落后 SupCon (-1.0/-1.6)。SupCon 的优势在 3-view 下也存在。
Remote exp191 ep110: 63.1/75.3 (OA-SD+CE ±0/+1.4 vs exp166 final!)
**决策**: 继续
