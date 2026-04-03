# exp230 Small + BT-PKD(0.01) + OA-SD 监控

配置: Small + GCN+PAA+OA-SD+PLBOA+ROA + BT-PKD(w=0.01)
对照: exp206r (Small OA-SD): 70.6/82.6 (eq), 72.3/82.9 (maxsim)
**注意**: 未使用 PARALLEL_AUG (3-view memory 不足 with BT-PKD non-detached graph)

## 检查点

### [03:48] 检查点 #1

刚启动。ep1 iter40。bt_pkd loss=0.532, oa_sd=0.482。
BT-PKD loss 正常：student 和 teacher 初始就不一致（不同的初始特征）。
**早停: ep10 < 30% 则终止。**
**决策**: 等 ep10 eval

### [04:00] 检查点 #2

ep7. Acc=0.167 — **正常！没有灾难！**
bt_pkd loss: 0.367→0.119 (快速下降，student 在对齐 teacher)
vs exp206r ep7: Acc ~0.15-0.17 — 一致。
**BT-PKD 的 cosine distillation 梯度没有破坏训练！**（vs BA-PKC SupCon 灾难 0.5%）
**决策**: 继续，密切监控 ep10 eval

### [04:06] 检查点 #3

ep10 训练中，eval 即将开始。Acc=0.188, bt_pkd=0.066 (持续下降)。
**决策**: 等 ep10 eval — 关键时刻

### [04:08] 检查点 #4 — ep10

**ep10: 48.4/60.9** (vs exp206r 50.4/63.9 = **-2.0/-3.0**)

没有灾难！远高于 30% 早停线。但落后 baseline -2.0/-3.0。
与 exp227 (GSPB+PADPQ) ep10 类似 (-2.2/-6.2)，后来追上并超过。
bt_pkd 梯度导致的轻微训练延迟 — 这与 GSPB 的模式一致。

**这是 Small 上首次非 detached 梯度存活！**
(BA-PKC: 0.5%, GSPB scale=0.01: 15.1%, GSPB scale=0.05: 2.3%)
BT-PKD w=0.01: 48.4% — **cosine distillation 梯度确实比 CE/SupCon 更温和。**

**决策**: 继续！观察 ep20-30 是否追上 baseline

### [04:12] 检查点 #5

ep12. Acc=0.191, bt_pkd=0.058 (持续下降，student-teacher 在对齐)。
正常训练节奏。ep20 eval ~16min。
**决策**: 继续

### [04:20] 检查点 #6

ep16. Acc=0.220, bt_pkd=0.055. ep20 eval ~8min。
**决策**: 等 ep20 eval

### [04:23] 检查点 #7

ep18. bt_pkd=0.056 (stabilized). ep20 eval ~4min.
**决策**: 等 ep20 eval

### [04:29] 检查点 #8 — ep20

**ep20: 57.3/68.6** (vs exp206r 56.6/68.1 = **+0.7/+0.5**)

| Epoch | exp230 mAP/R1 | exp206r mAP/R1 | delta |
|-------|------|------|------|
| 10 | 48.4/60.9 | 50.4/63.9 | -2.0/-3.0 |
| **20** | **57.3/68.6** | **56.6/68.1** | **+0.7/+0.5** |

**从 ep10 的 -2.0 到 ep20 的 +0.7 — 完全追上并超过！**
**mAP 和 R1 同时正向！**（PADPQ 只有 mAP 正向，R1 始终负向）
BT-PKD 的 cosine distillation 梯度在 Small 上有效！
**决策**: 继续！密切监控趋势是否持续

### [04:30] 检查点 #9 — GPU CRASH

**CUBLAS_STATUS_EXECUTION_FAILED** during/after ep20 eval.
Crash in triplet distance computation (euclidean_dist matmul).
可能原因: BT-PKD 的非 detached computation graph 占用额外内存 + eval → OOM → CUBLAS crash.
GPU 进入 error state, nvidia-smi 报 "Unknown Error"。需要系统重启。

ep20 checkpoint 已保存。ep20 结果已记录 (57.3/68.6 = **+0.7/+0.5 vs baseline**)。

### [04:57] 检查点 #10 — 重启

用户重启系统后 GPU 恢复。从头重新训练 (train.py 不支持 auto-resume)。
**新增 TEST.IMS_PER_BATCH 128** 防止 eval OOM。
第一轮 ep20=+0.7/+0.5 已确认 BT-PKD 有效，本次为完整 120ep 运行。
**决策**: 继续

### [05:08] 检查点 #11 — 第二次重跑 (无 PARALLEL_AUG)

第一次重跑无 PARALLEL_AUG: ep10=31.8/46.2 (正常，但不可与有 PARALLEL_AUG 的 baseline 比)。
第二次尝试加 PARALLEL_AUG: OOM! BT-PKD 的非 detached graph + 3-view 超过 24GB。
**最终方案**: 无 PARALLEL_AUG 运行到 final。比较需要找非 PARALLEL_AUG baseline。

### [05:19] 检查点 #12 — 第三次启动

ep1. 无 PARALLEL_AUG + TEST.IMS_PER_BATCH 128。正常启动。
ETA ~3h30m。
**决策**: 继续

### [05:27] 检查点 #13

ep4. bt_pkd=0.325 (比 OA-SD 版高因为没有 PARALLEL_AUG 的多 view 信号)。
正常训练。ep10 eval ~12min。
**决策**: 继续

### [05:30] 检查点 #14

ep6. Acc=0.126, bt_pkd=0.189。ep10 eval ~8min。
**决策**: 等 ep10 eval

### [05:32] 检查点 #15

ep7. Acc=0.198, bt_pkd=0.133. ep10 eval ~5min。
**决策**: 等 ep10 eval

### [05:35] 检查点 #16

ep9. Acc=0.158, bt_pkd=0.089. ep10 eval ~2min。
**决策**: 等 ep10 eval

### [05:40] 检查点 #17 — ep10

**ep10: 49.1/62.4** (无 PARALLEL_AUG)

对比第一轮 (有 PARALLEL_AUG): 48.4/60.9 — 基本一致。
eval 成功完成，TEST.IMS_PER_BATCH=128 解决了 OOM。
**注意**: 本次无 PARALLEL_AUG，需要找无 PARALLEL_AUG 的 baseline 对比。
ETA ~3h35m。
**决策**: 继续

### [05:48] 检查点 #18

ep14. bt_pkd=0.058. ep20 eval ~12min。
**决策**: 继续

### [05:50] 检查点 #19

ep16. bt_pkd=0.057. 正常收敛。ep20 eval ~8min。
**决策**: 继续

### [05:53] 检查点 #20

ep17. bt_pkd=0.056. ep20 eval ~6min。
**决策**: 等 ep20 eval

### [05:56] 检查点 #21

ep19. ep20 eval ~2min。
**决策**: 等 ep20 eval

### [06:01] 检查点 #22 — ep20

**ep20: 56.3/67.4** (无 PARALLEL_AUG)

对比第一轮 (有 PARALLEL_AUG): ep20=57.3/68.6。差异 -1.0/-1.2，符合 PARALLEL_AUG 的预期影响。
**eval 通过！TEST.IMS_PER_BATCH=128 解决了 OOM。**
ETA ~3h15m。
**决策**: 继续

### [06:10] 检查点 #23

ep25. bt_pkd=0.054 (稳定). ep30 eval ~10min。
**决策**: 等 ep30 eval

### [06:22] 检查点 #24 — ep30

**ep30: 62.6/74.2** (无 PARALLEL_AUG)
ETA ~3h。
**决策**: 继续

### [06:31] 检查点 #25

ep35. bt_pkd=0.054. ep40 eval ~10min。
**决策**: 继续

### [06:34] 检查点 #26

ep37. bt_pkd=0.056. ep40 eval ~6min。
**决策**: 等 ep40 eval

### [06:38] 检查点 #27

ep38. ep40 eval ~4min。
**决策**: 等 ep40 eval

### [06:40] 检查点 #28

ep40 iter40. eval 即将开始。
**决策**: 等 ep40 eval

### [06:43] 检查点 #29 — ep40

**ep40: 65.3/76.1** (无 PARALLEL_AUG)

| Epoch | exp230 mAP/R1 (无PAUG) |
|-------|------|
| 10 | 49.1/62.4 |
| 20 | 56.3/67.4 |
| 30 | 62.6/74.2 |
| **40** | **65.3/76.1** |

正常收敛曲线。需要对照无 PARALLEL_AUG 的 baseline 才能判断 BT-PKD 贡献。
ETA ~2h40m。
**决策**: 继续

### [06:54] 检查点 #30

ep46. bt_pkd=0.055. ep50 eval ~8min。
**决策**: 等 ep50 eval

### [06:57] 检查点 #31

ep47. ep50 eval ~5min。
**决策**: 等 ep50 eval

### [07:00] 检查点 #32

ep49. ep50 eval ~3min。
**决策**: 等 ep50 eval

### [07:05] 检查点 #33 — ep50

**ep50: 67.8/79.0** (无 PARALLEL_AUG)

| Epoch | exp230 mAP/R1 |
|-------|------|
| 10 | 49.1/62.4 |
| 20 | 56.3/67.4 |
| 30 | 62.6/74.2 |
| 40 | 65.3/76.1 |
| **50** | **67.8/79.0** |

正常收敛。ETA ~2h20m。
**决策**: 继续

### [07:17] 检查点 #34

ep57. bt_pkd=0.056. ep60 eval ~6min。
**决策**: 等 ep60 eval

### [07:20] 检查点 #35

ep58. ep60 eval ~4min。
**决策**: 等 ep60 eval

### [07:26] 检查点 #36 — ep60

**ep60: 67.7/78.8** (vs ep50: 67.8/79.0 — 持平，轻微下降)
这是中期平台期，正常。后期应该继续上升。
ETA ~2h。
**决策**: 继续

### [07:36] 检查点 #37

ep66. bt_pkd=0.055. ep70 eval ~8min。
**决策**: 继续

### [07:39] 检查点 #38

ep67. ep70 eval ~5min。
**决策**: 等 ep70 eval

### [07:42] 检查点 #39

ep69. ep70 eval ~2min。
**决策**: 等 ep70 eval

### [07:47] 检查点 #40 — ep70

**ep70: 69.0/80.3** (无 PARALLEL_AUG)

| Epoch | exp230 mAP/R1 |
|-------|------|
| 10 | 49.1/62.4 |
| 20 | 56.3/67.4 |
| 30 | 62.6/74.2 |
| 40 | 65.3/76.1 |
| 50 | 67.8/79.0 |
| 60 | 67.7/78.8 |
| **70** | **69.0/80.3** |

ep50-60 平台期后开始上升。正常收敛。
ETA ~1h40m。
**决策**: 继续

### [07:57] 检查点 #41

ep76. bt_pkd=0.056. ep80 eval ~8min。
**决策**: 继续

### [08:00] 检查点 #42

ep77. ep80 eval ~5min。
**决策**: 等 ep80 eval

### [08:03] 检查点 #43

ep79. ep80 eval ~2min。
**决策**: 等 ep80 eval

### [08:06] 检查点 #44

ep80 iter160. eval imminent.
**决策**: 等 ep80 eval

### [08:08] 检查点 #45 — ep80

**ep80: 70.3/81.1** (无 PARALLEL_AUG)

| Epoch | exp230 mAP/R1 |
|-------|------|
| 10 | 49.1/62.4 |
| 30 | 62.6/74.2 |
| 50 | 67.8/79.0 |
| 70 | 69.0/80.3 |
| **80** | **70.3/81.1** |

正常上升。ETA ~1h20m。
**决策**: 继续

### [08:22] 检查点 #46

ep87. ETA ~1h5m。ep90 eval ~6min。
**决策**: 等 ep90 eval

### [08:25] 检查点 #47

ep89. ep90 eval ~2min。
**决策**: 等 ep90 eval

### [08:28] 检查点 #48

ep90 iter180. eval imminent (~30s).
**决策**: 等 ep90 eval

### [08:29] 检查点 #49 — ep90

**ep90: 70.3/81.5** (= ep80 mAP, R1 +0.4)

| Epoch | exp230 mAP/R1 |
|-------|------|
| 50 | 67.8/79.0 |
| 70 | 69.0/80.3 |
| 80 | 70.3/81.1 |
| **90** | **70.3/81.5** |

ep80-90 平台期。预计 final ~71.0/81.5。
ETA ~58min。
**决策**: 继续

### [08:32] 检查点 #50

ep92. ep100 eval ~16min。
**决策**: 继续

### [08:39] 检查点 #51

ep95. ep100 eval ~10min。
**决策**: 等 ep100 eval

### [08:42] 检查点 #52

ep97. ep100 eval ~6min。
**决策**: 等 ep100 eval

### [08:45] 检查点 #53

ep98. ep100 eval ~4min。
**决策**: 等 ep100 eval

### [08:51] 检查点 #54 — ep100

**ep100: 70.5/81.2** (无 PARALLEL_AUG)

| Epoch | exp230 mAP/R1 |
|-------|------|
| 50 | 67.8/79.0 |
| 70 | 69.0/80.3 |
| 80 | 70.3/81.1 |
| 90 | 70.3/81.5 |
| **100** | **70.5/81.2** |

接近 exp206r final (70.6/82.6, 有 PARALLEL_AUG)。差异 -0.1/-1.4。
PARALLEL_AUG 主要影响 R1。mAP 几乎持平。
预计 final ~71.0/81.5。ETA ~39min。
**决策**: 继续

### [08:53] 检查点 #55

ep102. ep110 eval ~16min。
**决策**: 继续

### [09:04] 检查点 #56

ep107. ep110 eval ~6min。ETA ~25min。
**决策**: 等 ep110 eval

### [09:09] 检查点 #57

ep109. ep110 eval ~2min。
**决策**: 等 ep110 eval

### [09:14] 检查点 #58 — ep110

**ep110: 70.8/81.9** (无 PARALLEL_AUG)

| Epoch | exp230 mAP/R1 |
|-------|------|
| 80 | 70.3/81.1 |
| 90 | 70.3/81.5 |
| 100 | 70.5/81.2 |
| **110** | **70.8/81.9** |

还在上升。预计 final ~71.0/82.0。
ETA ~19min。
**决策**: 继续到 final

### [09:21] 检查点 #59 — OOM CRASH after ep110

**OOM at ep120 eval** — CUDA out of memory。
Training speed 从 118 s/ep 降到 140 s/ep (ep107+) — memory 持续积累。
BT-PKD 的非 detached graph 在 Small 上内存不够。

**最终可用结果: ep110 = 70.8/81.9** (无 PARALLEL_AUG)

## 结论

exp230 (Small BT-PKD, w=0.01, constant, no PARALLEL_AUG):
- ep10: 49.1/62.4
- ep110: **70.8/81.9** (最后可用结果)
- 预计 final ≈ 71.0/82.0

vs exp206r (Small OA-SD, PARALLEL_AUG): 70.6/82.6
差异: **+0.2/-0.7** (mAP 基本持平, R1 -0.7 主要因缺 PARALLEL_AUG)

**BT-PKD 在 Small 上基本中性** — 没有明显正面也没灾难。
但 **BT-PKD + PARALLEL_AUG 的内存问题需要解决** 才能正确比较。
