# exp203 Swin-Small + GCN+PAA+ROA + SupCon + PLBOA + 3-view 监控

配置: pose_psg_gcn_paa_roa.yml + Small + SupCon T=0.05 + PLBOA 0.7 + 3-view + WITH_CP
对照:
- 4090 PAA (Small, GCN+PAA+ROA, CE): **70.8/81.7**
- exp202b (Small, STD-PR+SupCon+3-view): 69.3/80.2

**目标**: 超过 70.8/81.7！

## 检查点

### [04:40] 检查点 #1

**状态**: 正常
**进度**: Epoch 1/120

| 指标 | 当前值 |
|------|--------|
| supcon | 3.975 |
| id_global | 6.554 |
| tri_global | 15.793 |
| tri_part | 16.912 (GCN part triplet, 比 STD-PR 高) |
| GPU | **6.8GB/24GB** (GCN 很轻量！) |

**观察**: GCN+PAA+SupCon+3-view 成功启动。GPU 仅 6.8GB！
比 STD-PR (8.4GB with CP) 还少。GCN 架构更高效。
**决策**: 继续
