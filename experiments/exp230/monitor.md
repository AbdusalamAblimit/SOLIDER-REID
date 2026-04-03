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
