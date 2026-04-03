# exp229 Tiny + BT-PKD(0.01) + OA-SD 监控

配置: Tiny + GCN+PAA+OA-SD+PLBOA+ROA + BT-PKD(w=0.01)
对照: exp191 (Tiny OA-SD): 63.2/75.4
**创新**: 非 detached 的 per-keypoint cosine distillation 到 backbone

## 检查点

### [12:00] 检查点 #1

远程刚启动。ep1 done. bt_pkd=0.48, oa_sd=0.40。
BT-PKD loss 正常 — student 和 teacher 刚开始对齐。
ETA ~3h58m。
**早停: ep10 < 25% 则终止。**
**决策**: 等 ep10 eval

### [12:09] 检查点 #2

ep6. Acc=0.135, bt_pkd=0.204。正常训练。
vs OA-SD-only (exp191) ep6: Acc ~0.12 — **BT-PKD 与 baseline 一致或略好！**
ep10 eval ~8min。
**决策**: 继续
