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

### [10:08] 检查点 #6

ep19. bt_pkd=0.053. ep20 eval ~2min。
**决策**: 等 ep20 eval

### [10:10] 检查点 #7

ep20 iter100. eval ~1min。
**决策**: 等 ep20 eval

### [10:13] 检查点 #8 — ep20

**ep20: 55.7/67.7** (vs exp230 constant 56.3/67.4 = -0.6/+0.3)
接近 constant 版。decay 在 ep20 尚未显著影响。
ETA ~3h15m。
**决策**: 继续

### [10:18] 检查点 #9

ep23. bt_pkd=0.052. ep30 eval ~14min。
**决策**: 继续

### [10:29] 检查点 #10

ep28. bt_pkd=0.055. ep30 eval ~4min。
**决策**: 等 ep30 eval

### [10:32] 检查点 #11

ep29. ep30 eval ~2min。
**决策**: 等 ep30 eval

### [10:36] 检查点 #12 — ep30

**ep30: 61.3/72.8** (vs exp230 constant 62.6/74.2 = -1.3/-1.4)
低于 constant 版。与 Tiny (exp231) 的模式类似 — decay 在早期稍低。
ETA ~2h55m。
**决策**: 继续
