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
