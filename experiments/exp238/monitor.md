# exp238 Tiny + PPA (assign_weight=0.1) + OA-SD 监控

配置: 与 exp237 相同，仅 POSE_PPA_ASSIGN_WEIGHT=0.1 (vs exp237 的 0.5)
对照: exp237 (PPA w=0.5): 63.7/75.0 (+0.5/-0.4)
对照: exp191 (detached GCN): 63.2/75.4

## 检查点

### [10:14] 检查点 #1

远程启动。ep1. ppa_assign=1.62. ETA ~4h9m。
**决策**: 等 ep10 eval

### [10:35] 检查点 #2

### [10:35] 检查点 #3 — ep10

**ep10: 40.4/52.4** (vs exp191 34.3/46.8 = **+6.1/+5.6**)

**PPA w=0.1 比 w=0.5 (exp237 +2.2/+1.0) 强得多！**
assign_weight=0.1 让 assignment gradient 更温和 → backbone 更好地融合。
这是所有实验中最强的 ep10 结果！

| 实验 | ep10 delta |
|------|------|
| exp237 PPA w=0.5 | +2.2/+1.0 |
| **exp238 PPA w=0.1** | **+6.1/+5.6** |
| FSDC exp236 | +4.6/+6.3 |
| BT-PKD exp229 | +3.2/+3.2 |

**决策**: 密切监控！如果 final 也正向，这就是突破
