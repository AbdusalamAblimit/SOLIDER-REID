# exp238 Tiny + PPA (assign_weight=0.1) + OA-SD 监控

配置: 与 exp237 相同，仅 POSE_PPA_ASSIGN_WEIGHT=0.1 (vs exp237 的 0.5)
对照: exp237 (PPA w=0.5): 63.7/75.0 (+0.5/-0.4)
对照: exp191 (detached GCN): 63.2/75.4

## 检查点

### [10:14] 检查点 #1

远程启动。ep1. ppa_assign=1.62. ETA ~4h9m。
**决策**: 等 ep10 eval
