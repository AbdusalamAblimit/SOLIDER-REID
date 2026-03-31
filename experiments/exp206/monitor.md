# exp206 Swin-Small + GCN+PAA + CE + OA-SD (远程 1-view) 监控

配置: Swin-Small + GCN+PAA+ROA + CE + OA-SD (decay=0.999) + PLBOA
对照: 4090 PAA (GCN+PAA, CE, no OA-SD): **70.8/81.7**

**目标**: 4090 PAA 70.8 + OA-SD (+2.9 on Tiny) → **72-73%+**

## 检查点

### [22:50] 检查点 #1

**状态**: 正常
**进度**: Epoch 1/120, ETA 5h25m

| 指标 | 当前值 |
|------|--------|
| oa_sd | 0.480 |
| id_global | 6.554 |
| tri_part | 11.865 (GCN part triplet) |
| Speed | 81.9 samples/s |

**观察**: OA-SD+GCN+PAA 成功启动。oa_sd=0.48 正常。
**决策**: 继续
