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

### [12:17] 检查点 #3

ep10 训练中。Acc=0.125, bt_pkd=0.112。
ep10 eval ~2min。
**决策**: 等 ep10 eval

### [12:19] 检查点 #4 — ep10

**ep10: 37.5/50.0** (vs exp191 OA-SD ep10: 34.3/46.8 = **+3.2/+3.2**)

**BT-PKD 在 Tiny 上正向！** 超过 baseline +3.2/+3.2。
对比 GSPB ep10 (+5.8)，BT-PKD 也在加速早期收敛，但幅度较小。
ETA ~3h36m。
**决策**: 继续！密切监控

### [12:23] 检查点 #5

ep12. Acc=0.173, bt_pkd=0.098. ep20 eval ~16min.
**决策**: 继续

### [12:30] 检查点 #6

ep16. Acc=0.223, bt_pkd=0.105. ep20 eval ~8min.
**exp230 (Small BT-PKD) ep20: +0.7/+0.5 vs baseline — 正向！**
**决策**: 等 ep20 eval

### [12:41] 检查点 #7 — ep20

**ep20: 47.5/58.6** (vs exp191 46.0/58.0 = **+1.5/+0.6**)

| Epoch | exp229 mAP/R1 | exp191 mAP/R1 | delta |
|-------|------|------|------|
| 10 | 37.5/50.0 | 34.3/46.8 | +3.2/+3.2 |
| **20** | **47.5/58.6** | **46.0/58.0** | **+1.5/+0.6** |

早期加速在收敛 (+3.2→+1.5)。与 GSPB 模式类似 (ep10 +5.8 → 后来持平)。
但 R1 也正向 (+0.6) — 比 PADPQ 好 (PADPQ R1 始终负)。
ETA ~3h17m。
**决策**: 继续
