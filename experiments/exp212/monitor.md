# exp212 Small + GCN+PAA+CE+OA-SD LR=0.0008 监控

配置: exp206r + LR=0.0008 (vs exp206r LR=0.0004)
对照: exp206r (70.6/82.6 eq, 72.3/82.9 maxsim)

## 检查点

### [00:20] 检查点 #1

**状态**: 正常
**进度**: Epoch 1/120

| 指标 | 当前值 |
|------|--------|
| oa_sd | 0.573 |
| id_global | 6.554 |
| tri_global | 16.1 |

LR=0.0008 启动正常。oa_sd=0.573 (略高于 exp206r 初始的 0.448——因为 LR 更高，warmup 阶段 teacher/student 差距更大)。
**决策**: 继续

### [00:28] 检查点 #2

ep5. id_global=6.549, Acc=0.003. oa_sd=0.015.
正常 warmup 阶段（与 exp206r 初期类似）。
ep10 eval ~15min.
**决策**: 继续

### [00:33] 检查点 #3

ep8. id_global=6.532, Acc=0.012, oa_sd=0.006.
ep10 eval ~6min.
**决策**: 等 eval

### [00:39] ep10 — 终止！

**ep10: 0.8/1.3% — 灾难！LR=0.0008 对 Small 太高！**

模型完全没学到。id_global 仅从 6.554 降到 6.532 (ep8)。
oa_sd=0.006 说明 teacher≈student, 但两者都没在学。

**结论: Small 需要 LR=0.0004，不能用 Tiny 的 0.0008。**
**实验终止。本地 GPU 空闲。**
