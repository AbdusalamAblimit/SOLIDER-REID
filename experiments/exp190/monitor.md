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
**决策**: 继续到 ep80 看最终是否 CE vs SupCon 有分化
