# exp232 Small + BT-PKD cosine decay (w→0 by ep60) 监控

配置: Small + GCN+PAA+OA-SD+PLBOA+ROA + BT-PKD(w=0.01, decay_ep=60), 无 PARALLEL_AUG
对照: exp230 (Small BT-PKD constant, no PAUG): ep110=70.8/81.9
对照: exp206r (Small OA-SD, PAUG): 70.6/82.6

## 检查点

### [09:32] 检查点 #1

ep1. bt_pkd=0.531. 正常启动。
ETA ~3h30m。
**决策**: 等 ep10 eval

### [09:35] 检查点 #2

ep2. bt_pkd=0.658. 正常早期训练。ep10 eval ~16min。
**决策**: 继续

### [09:44] 检查点 #3

ep7. bt_pkd=0.088. 正常收敛。ep10 eval ~6min。
**决策**: 等 ep10 eval

### [09:47] 检查点 #4

ep9. bt_pkd=0.061. ep10 eval ~2min。
**决策**: 等 ep10 eval

### [09:52] 检查点 #5 — ep10

**ep10: 45.1/57.3** (vs exp230 constant 49.1/62.4 = -4.0/-5.1)
低于 constant 版，但 decay 在 ep10 几乎没影响 (w=0.0098)。
差异可能来自训练随机性。
**决策**: 继续
