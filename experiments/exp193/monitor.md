# exp193 OA-SD + 3-view Parallel Aug + CE 监控

配置: exp190 (3-view+CE) + POSE_OA_SD=True (EMA teacher, decay=0.999)
对照:
- exp190 (3-view+CE): 64.2/75.6
- exp191 (OA-SD+CE, 1-view): 63.2/75.4
- exp187 (3-view+SupCon): 64.9/76.6

## 检查点

### [04:36] 检查点 #1

**状态**: 正常
**进度**: Epoch 1/120

| 指标 | 当前值 | 备注 |
|------|--------|------|
| id_global | 6.555 | 初始 |
| id_part | 6.696 | 初始 |
| tri_global | 12.19 | 下降中 |
| oa_sd | 0.372 | 正常范围 (exp191 ep1=0.562, exp192 ep1=0.446) |
| GPU Memory | 20.7GB/24GB | 安全 (+1.2GB vs exp190) |
| Speed | — | 待确认 |

**观察**: OA-SD + 3-view 成功启动，无 OOM。oa_sd=0.37 在合理范围。

### [04:38] 检查点 #2

**状态**: 正常
**进度**: Epoch 2/120

| 指标 | 当前值 | 趋势 |
|------|--------|------|
| Total Loss | 9.832 | ↓ 快速下降 |
| id_global | 6.552 | 初始 |
| id_part | 6.702 | 初始 |
| tri_global | 4.494 | ↓ 快速 |
| tri_part | 0.771 | ↓ |
| oa_sd | 0.484 | ↑ 从 0.37→0.49 (teacher 落后于 student 正常现象) |
| str_token_norm | 97.6 | ↓ 从 102→98 |
| Speed | 87.8 samples/s | 正常 (比 exp190 慢 ~7% 因 teacher forward) |
| ETA | 5h03m | — |

**观察**: oa_sd=0.49 比 exp191 ep2 (0.56) 低一些，比 exp192 ep2 (0.35) 高。
speed 比 3-view only 慢 ~7%（87.8 vs 94.5），符合预期。
**决策**: 继续
