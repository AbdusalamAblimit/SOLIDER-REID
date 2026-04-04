# exp242 Small + PPA + GCN 双分支 + OA-SD 监控

配置: Small + PSG + PPA + GCN + OA-SD + PLBOA(0.7) + 无 ROA
对照: exp206r (Small GCN): 70.6/82.6

## 检查点

### [22:53] 检查点 #1

远程启动。ep1. ppa_assign=1.56. 未 OOM。
**决策**: 等 ep10 eval

### [23:05] 检查点 #2

ep5. ppa_assign=0.69. ep10 eval ~10min。
**决策**: 等 ep10 eval

### 最终结果 (ep120)

训练在 context compact 期间完成。从 remote log 获取完整结果。

| Epoch | mAP | R1 | R10 |
|-------|-----|-----|-----|
| 10 | 38.8 | 51.3 | 73.1 |
| 20 | 47.0 | 59.6 | 80.6 |
| 30 | 53.1 | 64.8 | 83.8 |
| 40 | 55.4 | 67.1 | 86.4 |
| 50 | 57.6 | 70.5 | 86.4 |
| 60 | 58.0 | 70.5 | 86.5 |
| 70 | 59.8 | 71.3 | 88.0 |
| 80 | 60.1 | 72.0 | 87.8 |
| 90 | 60.4 | 73.3 | 88.5 |
| 100 | 60.7 | 72.9 | 88.7 |
| 110 | 60.9 | 73.1 | 88.7 |
| **120** | **60.9** | **73.4** | **88.9** |

## 结论

**exp242 FINAL: 60.9/73.4** vs exp206r (Small GCN baseline): 70.6/82.6 = **-9.7/-9.2**

**灾难性失败！** PPA+GCN 在 Small 上完全不工作。

分析：
- PPA 的 non-detached 梯度严重损害了更强的 Small backbone
- Small 的 backbone 已经非常优化，额外的 part assignment 梯度造成灾难性干扰
- 对比 Tiny: PPA+GCN +0.5/-0.1，Small: -9.7/-9.2
- **PPA 方法不可泛化到更大 backbone** — 这是根本性问题

**关键发现**: non-detached part gradients 对 backbone 的影响与模型大小负相关。Small backbone 的特征空间更结构化，PPA 的粗粒度梯度破坏了这个结构。
