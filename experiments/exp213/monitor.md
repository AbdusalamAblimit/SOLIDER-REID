# exp213 Small + GCN+PAA+CE+OA-SD + PKC(0.05) + MST(0.1) 监控

配置: exp206r + PKC weight=0.05 + MST weight=0.1
对照: exp206r (72.3 maxsim), exp210b PKC-only (72.4 maxsim), exp211 MST-only (TBD)

## 检查点

### [00:48] 检查点 #1

**状态**: 正常
**进度**: Epoch 1/120

| 指标 | 当前值 |
|------|--------|
| oa_sd | 0.411 |
| pkc | 3.599 |
| mst | 0.481 |
| pkc_nk | 17.0 |
| id_global | 6.554 |

两个 per-keypoint loss 同时工作。总额外贡献: 0.05*3.6 + 0.1*0.48 = 0.23 (vs 总 loss ~25)。
**决策**: 继续

### [00:53] 检查点 #2

ep4. pkc=3.135, mst=0.309 (两者都在下降). Speed 122.7 s/s, ETA 3h35m.
exp211 ep10: 50.4/63.9 (= exp206r, MST=0.5 无害!)
**决策**: 继续。exp213 ep10 eval ~17min.

### [00:59] 检查点 #3

ep7. pkc=3.285, mst=0.298. Acc=0.134.
ep10 eval ~8min.
**决策**: 等 eval

### [01:05] 检查点 #4

ep10 开始. mst=0.295, pkc=3.286. eval ~3min.
**决策**: 等 eval

### [01:08] ep10 — 终止！

**ep10: 40.6/54.8** (vs exp206r 50.4/63.9 = **-9.8/-9.1!**)

PKC+MST 结合灾难！单独都正常 (PKC=50.6, MST=50.4) 但结合后严重下降。
两个 loss 在 per-keypoint features 上有梯度冲突。
**实验终止。**

**结论: per-keypoint losses 只能单独使用，不能结合。**
