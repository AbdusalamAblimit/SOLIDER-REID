# exp240 Small + PPA (w=0.5) + OA-SD 监控

配置: Small + PSG + PPA (w=0.5) + OA-SD + PLBOA(0.7) + 无 ROA
对照: exp206r (Small OA-SD GCN): 70.6/82.6

## 检查点

### [06:57] 检查点 #1

本地启动。ep1. ppa_assign=1.71. 未 OOM。
**决策**: 等 ep10 eval

### [07:09] 检查点 #2

### [07:17] 检查点 #3 — ep10

**ep10: 49.2/61.8** (vs exp206r 50.4/63.9 = -1.2/-2.1)
但 exp206r 有 PARALLEL_AUG, exp240 没有。
vs exp230 (no PAUG) ep10=49.1 → PPA 基本持平。
**决策**: 继续
