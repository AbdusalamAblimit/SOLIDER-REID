# exp231 Tiny + BT-PKD cosine decay (w=0.01 → 0 by ep60) 监控

配置: Tiny + GCN+PAA+OA-SD+PLBOA+ROA + BT-PKD(w=0.01, decay_epoch=60)
对照: exp191 (Tiny OA-SD): 63.2/75.4
对照: exp229 (BT-PKD constant): 62.2/75.0 (-1.0/-0.4)

## 检查点

### [16:18] 检查点 #1

远程启动成功。ep1. bt_pkd=0.485。
BT-PKD weight schedule: ep1=0.01, ep30=0.005, ep45=0.0015, ep60=0.0。
ETA ~4h。
**决策**: 等 ep10 eval

### [16:22] 检查点 #2

ep3. bt_pkd=0.398. 正常训练。ep10 eval ~14min。
**决策**: 继续

### [16:25] 检查点 #3

ep5. bt_pkd=0.266. ep10 eval ~10min。
**决策**: 继续

### [16:28] 检查点 #4

ep6. bt_pkd=0.205. ep10 eval ~8min。
**决策**: 等 ep10 eval

### [16:32] 检查点 #5

ep8. ep10 eval ~4min。ETA ~3h43m。
**决策**: 等 ep10 eval

### [16:35] 检查点 #6

ep10 iter80. eval ~1min。
**决策**: 等 ep10 eval

### [16:38] 检查点 #7 — ep10

**ep10: 38.6/51.5** (vs exp191 34.3/46.8 = **+4.3/+4.7**, vs exp229 37.5/50.0 = +1.1/+1.5)

BT-PKD decay 的早期加速与 constant 一样强（decay 在 ep10 几乎没开始: w=0.0098）。
**关键测试**: ep60+ 时 BT-PKD 已关闭 (w=0), backbone 是否能继续正常收敛？
ETA ~3h36m。
**决策**: 继续！密切监控 ep50-70 区间

### [16:42] 检查点 #8

ep13. bt_pkd=0.102. 正常训练。ep20 eval ~14min。
**决策**: 继续

### [16:53] 检查点 #9

ep18. bt_pkd=0.116 (注意: bt_pkd 反而升了，说明 decay 还未显著生效)。
BT-PKD weight at ep18: w=0.01 * 0.5*(1+cos(π*18/60)) ≈ 0.01*0.81 = 0.0081。
ep20 eval ~4min。
**决策**: 等 ep20 eval

### [16:56] 检查点 #10

ep20 开始。bt_pkd=0.128。eval ~2min。
**决策**: 等 ep20 eval

### [16:59] 检查点 #11 — ep20

**ep20: 46.3/58.6** (vs exp191 46.0/58.0 = **+0.3/+0.6**, vs exp229 47.5/58.6 = -1.2/0.0)

| Epoch | exp231 (decay) | exp229 (const) | exp191 (base) | delta vs base |
|-------|------|------|------|------|
| 10 | 38.6/51.5 | 37.5/50.0 | 34.3/46.8 | **+4.3/+4.7** |
| **20** | **46.3/58.6** | **47.5/58.6** | **46.0/58.0** | **+0.3/+0.6** |

Decay 版略低于 constant 版 (-1.2 mAP at ep20)。decay weight ~0.0075 vs constant 0.01。
关键转折点在 ep50-60：constant 版在此下降，decay 版应该不会。
ETA ~3h20m。
**决策**: 继续！关键在 ep50+ 的对比
