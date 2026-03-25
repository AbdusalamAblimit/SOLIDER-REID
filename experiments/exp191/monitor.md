# exp191 OA-SD + CE 监控

配置: exp166 + POSE_OA_SD=True (EMA teacher + CE loss, 无 SupCon)

## 检查点

### [05:01] 检查点 #1

**状态**: 正常
**进度**: Epoch 2/120

| 指标 | 当前值 | 趋势 |
|------|--------|------|
| oa_sd | 0.562 | ↑ 上升（初期正常） |
| id_global | 6.554 | 初始 |
| id_part | 6.704 | 初始 |
| Tri Global | 6.4 | 快速下降 |
| 速度 | ~90 samples/s | 正常（2x forward） |

**观察**: OA-SD 在 CE 环境下正常工作。oa_sd loss 0.56 比 exp188 (SupCon) 的 0.48 略高。
**决策**: 继续
