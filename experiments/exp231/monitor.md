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
