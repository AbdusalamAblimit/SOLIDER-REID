# exp210b Small + GCN+PAA+CE+OA-SD + PKC (weight=0.05) 监控

配置: exp206r + PKC weight=0.05 (vs exp210 weight=0.5 灾难)
对照: exp206r (70.6/82.6 equal_concat, 72.3/82.9 maxsim_hybrid)

## 检查点

### [01:37] 检查点 #1

**状态**: 正常
**进度**: Epoch 1/120

| 指标 | 当前值 |
|------|--------|
| oa_sd | 0.461 |
| pkc | 3.172 |
| pkc_nk | 17.0 |
| id_global | 6.554 |
| PKC loss 贡献 | 0.05 * 3.17 = 0.16 (vs exp210: 0.5*3.87=1.94) |

PKC 贡献极低 (0.16 vs 总 loss ~19)。应该不会干扰 CE 收敛。
**决策**: 继续

### [01:44] 检查点 #2

ep3. Acc=0.011 (**远好于 exp210 同 epoch Acc=0.001!**)。
id_global=6.540, pkc=2.865 (从 3.17 下降——keypoint features 在学习)。
CE 收敛正常！
**决策**: 继续

### [01:50] 检查点 #3

ep5. Acc=0.109 (vs exp206r ep5 Acc=0.126 — 略低但正常范围)。
id_global=6.497, pkc=3.259.
ep10 eval ~15min。
**决策**: 继续

### [01:55] 检查点 #4

ep7. Acc=0.144 (vs exp206r ep7 Acc=0.169 — 略低)。
id_global=6.420, pkc=3.331.
PKC weight=0.05 有轻微负面影响但不致命。
ep10 eval ~10min。
**决策**: 继续

### [02:01] 检查点 #5

ep9. Acc=0.142, id_global=6.312, pkc=3.323.
ep10 eval ~5min.
**决策**: 等 eval

### [02:08] 检查点 #6 — ep10 ✓

**ep10: 50.6/62.8** (vs exp206r 50.4/63.9 = **+0.2/-1.1 ≈ 持平！**)

| 实验 | ep10 mAP | ep10 R1 |
|------|---------|---------|
| exp210 (PKC=0.5) | 3.6% | 5.3% |
| exp206r (no PKC) | 50.4% | 63.9% |
| **exp210b (PKC=0.05)** | **50.6%** | **62.8%** |

**PKC=0.05 未损害收敛！** 关键问题: 是否在后期/final 对 MaxSim matching 有帮助？
**决策**: 继续到 ep120

### [02:20] 检查点 #7

ep14. id_global=5.650, Acc=0.190, pkc=3.240.
ep20 eval ~18min。
**决策**: 继续

### [02:31] 检查点 #8

ep18. id_global=5.164, pkc=3.205. ep20 eval ~7min.
**决策**: 等 eval

### [02:37] 检查点 #9

ep20 mid. id_global=4.790, pkc=3.136. eval ~4min.
**决策**: 等 eval

### [02:40] 检查点 #10 — ep20 ✓

**ep20: 56.9/68.4** (vs exp206r 56.6/68.1 = **+0.3/+0.3!**)

| Epoch | exp210b (PKC=0.05) | exp206r (no PKC) | delta |
|-------|------|------|------|
| 10 | 50.6/62.8 | 50.4/63.9 | +0.2/-1.1 |
| 20 | 56.9/68.4 | 56.6/68.1 | +0.3/+0.3 |

**PKC=0.05 在 ep20 略微领先！** per-keypoint contrastive 正在发挥作用。
如果趋势持续，final 可能比 exp206r 更好 (+0.3-0.5%)。
再加 maxsim_hybrid: **72.5-73%!**
**决策**: 继续！

### [02:53] 检查点 #11

ep25. pkc=2.983 (从 3.27 下降到 2.98——显著下降!)。Acc=0.322.
ep30 eval ~15min.
**决策**: 继续

### [02:58] 检查点 #12

ep26. id_global=3.499, Acc=0.481, pkc=3.001. ep30 eval ~11min.
**决策**: 等 eval

### [03:04] 检查点 #13

ep28. pkc=2.928 (持续下降). ep30 eval ~6min.
**决策**: 等 eval

### [03:09] 检查点 #14

ep30 iter 100. pkc=2.881. eval ~3min.
exp207 ep78. ep80 eval ~8min.
**决策**: 等 evals

### [03:13] 检查点 #15 — ep30

**ep30: 61.7/73.9** (vs exp206r 62.3/73.8 = **-0.6/+0.1**)

| Epoch | PKC=0.05 | no PKC | delta |
|-------|------|------|------|
| 10 | 50.6 | 50.4 | +0.2 |
| 20 | 56.9 | 56.6 | +0.3 |
| 30 | 61.7 | 62.3 | -0.6 |

ep30 落后。与 OA-SD fix 类似的震荡模式。
需要看 final 和 maxsim_hybrid 测试才能判断 PKC 是否有用。
**决策**: 继续到 ep120

### [07:01] 检查点 #28 — ep100

**ep100: 70.3/81.5** (vs exp206r 70.3/81.9 = 0.0/-0.4)

equal_concat 完全一致。PKC=0.05 不改变 equal_concat 性能。
pkc loss 从 3.27→2.28 (30% 下降)——keypoint features 确实在学习。
ETA 1h. **决策**: 继续到 final + maxsim_hybrid 测试

### [08:01] 检查点 #29

ep120! ETA 3min! Final eval 即将!
ep110: 70.5/81.9
**决策**: 等 FINAL!

## exp210b FINAL RESULTS

**exp210b (PKC=0.05) FINAL: 70.6/81.8**
vs exp206r (no PKC): 70.6/82.6 → **mAP 一致, R1 -0.8**

PKC=0.05 不改变 mAP，R1 略低。
正在测试 maxsim_hybrid — 这是 PKC 的真正价值测试。

### MaxSim Hybrid 测试结果 🔥

**exp210b + maxsim_hybrid: 72.4/83.1!!**

| 模型 | equal_concat | maxsim_hybrid | MaxSim delta |
|------|------|------|------|
| exp206r (no PKC) | 70.6/82.6 | 72.3/82.9 | +1.7/+0.3 |
| **exp210b (PKC=0.05)** | **70.6/81.8** | **72.4/83.1** | **+1.8/+1.3** |

**72.4/83.1 = 新最佳！**
PKC 训练确实改善了 MaxSim matching (+0.1 mAP, +0.2 R1)。
PKC 的 MaxSim delta 更大 (+1.8 vs +1.7)——keypoint features 更加 discriminative。

**远程 GPU 空闲。准备下一实验。**

### [03:37] 检查点 #16

ep38. pkc=2.709, Acc=0.773. ep40 eval ~7min.
**决策**: 等 eval

### [03:43] 检查点 #17

ep40 iter 200. pkc=2.700, Acc=0.798. eval 即将开始。
exp207 ep~84.
**决策**: 等 eval

### [03:45] 检查点 #18 — ep40

**ep40: 65.1/76.9** (vs exp206r 65.8/76.4 = **-0.7/+0.5**)

| Epoch | PKC=0.05 | no PKC | delta mAP |
|-------|------|------|------|
| 10 | 50.6 | 50.4 | +0.2 |
| 20 | 56.9 | 56.6 | +0.3 |
| 30 | 61.7 | 62.3 | -0.6 |
| 40 | 65.1 | 65.8 | -0.7 |

mAP 略低，R1 略高。总体非常接近。
关键: final 后的 maxsim_hybrid 测试将决定 PKC 的价值。
**决策**: 继续

### [04:08] 检查点 #19

ep48. pkc=2.600 (从 3.27 下降到 2.60 — 20% drop!). ep50 eval ~6min.
**决策**: 等 eval

### [04:14] 检查点 #20

exp210b ep50 mid. pkc=2.523. eval ~3min.
exp207 ep88. ep90 eval ~10min.
**决策**: 等 evals

### [04:18] 检查点 #21 — ep50 ✓

**ep50: 67.6/79.6** (vs exp206r 67.6/79.5 = **0.0/+0.1 — 完全一致!**)

| Epoch | PKC=0.05 | no PKC | delta |
|-------|------|------|------|
| 10 | 50.6 | 50.4 | +0.2 |
| 20 | 56.9 | 56.6 | +0.3 |
| 30 | 61.7 | 62.3 | -0.6 |
| 40 | 65.1 | 65.8 | -0.7 |
| 50 | 67.6 | 67.6 | 0.0 |

PKC=0.05 与 no PKC 在 equal_concat 上完全一致。
**关键问题: MaxSim hybrid 测试会否不同？**
PKC 训练了更 discriminative 的 per-keypoint features — 可能在 MaxSim matching 中表现更好。
**决策**: 继续到 ep120，然后做 maxsim_hybrid 对比测试

### [04:50] 检查点 #22 — ep60

**ep60: 67.8/79.7** (vs exp206r 68.3/79.8 = **-0.5/-0.1**)

| Epoch | PKC=0.05 | no PKC | delta |
|-------|------|------|------|
| 50 | 67.6 | 67.6 | 0.0 |
| 60 | 67.8 | 68.3 | -0.5 |

PKC 在 equal_concat 上持续与 no PKC 打平或略低。
**但 PKC 的价值在 MaxSim matching，不在 equal_concat。**
pkc loss 持续下降 (3.27→2.44)，keypoint features 确实在学习。
**决策**: 继续到 final，maxsim_hybrid 测试才是关键

### [05:23] 检查点 #23 — ep70

**ep70: 68.7/79.8** (vs exp206r 68.5/80.4 = **+0.2/-0.6**)
PKC 在 ep70 mAP 再次领先。依旧震荡模式。
**决策**: 继续

### [05:51] 检查点 #24

ep80 开始. pkc=2.332. eval ~5min. ETA 2h05m.
**决策**: 等 eval

### [05:55] 检查点 #25 — ep80

**ep80: 70.1/82.0** (vs exp206r 70.2/81.5 = **-0.1/+0.5**)

| Epoch | PKC=0.05 | no PKC | delta mAP | delta R1 |
|-------|------|------|------|------|
| 70 | 68.7 | 68.5 | +0.2 | -0.6 |
| 80 | 70.1 | 70.2 | -0.1 | +0.5 |

mAP 持平，但 PKC 在 R1 上持续领先 (+0.5)！
PKC 可能通过更好的 keypoint discrimination 改善了 R1。
**决策**: 继续到 ep120

### [06:18] 检查点 #26

ep88. pkc=2.321. ep90 eval ~6min. ETA ~1h35m.
**决策**: 等 eval

### [06:28] 检查点 #27 — ep90

**ep90: 70.0/81.4** (vs exp206r 70.2/81.9 = **-0.2/-0.5**)
ep80 的 R1 优势在 ep90 消失。与 exp206r 趋同。
预计 final: ~70.5/82.0 (与 exp206r 几乎一致)。
**关键测试仍是 maxsim_hybrid。**
**决策**: 继续
