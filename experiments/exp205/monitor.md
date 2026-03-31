# exp205 Dual Part Branch: GCN+PAA + STD-PR SupCon on Swin-Small 监控

配置: Swin-Small + GCN+PAA+ROA + STD-PR per-token SupCon + PLBOA + 3-view + WITH_CP
对照:
- 4090 PAA (GCN+PAA, CE): **70.8/81.7**
- exp202b (STD-PR+SupCon, 3-view): 69.3/80.2
- exp203 (GCN+PAA+SupCon, 3-view): ep20=57.0/68.7

**目标**: 超过 70.8！GCN 架构 + STD-PR per-token SupCon 双分支。

## 检查点

### [06:24] 检查点 #1

**状态**: 正常
**进度**: Epoch 1/120

| 指标 | 当前值 | 备注 |
|------|--------|------|
| supcon | 4.665 | STD-PR per-token SupCon ✓ |
| str_token_norm | 103.3 | STD-PR tokens 在运行 ✓ |
| tri_part | 0.821 | GCN part triplet ✓ |
| GPU | **7.2GB/24GB** | 双分支+CP 非常高效 |

**观察**: 双分支成功！GCN 和 STD-PR 同时工作。GPU 仅 7.2GB。
### [06:25] 检查点 #2

Dual branch 正常运行。supcon=4.386 (STD-PR), tri_part=0.806 (GCN)。
### [06:31] 检查点 #3

ep2/120. supcon=4.141, tri_part=0.792. 训练正常。
### [06:36] 检查点 #4

ep3/120. tri_global=2.3 (↓↓ 快速下降), tri_part=0.734。
### [06:42] 检查点 #5

### [06:48] GPU 崩溃 ⚠️

exp205 在 ep4 附近 CUDA 错误崩溃。`CUBLAS_STATUS_EXECUTION_FAILED`。
nvidia-smi 返回 "Unknown Error" — 与之前 exp193 相同的 GPU 硬件/驱动问题。
**需要重启容器/机器恢复 GPU。**

### [09:43] GPU 恢复，exp205 重启

**本地 exp205**: 3-view + CP 重启成功。7.2GB/24GB。ETA ~9h。
**远程 exp205r**: 1-view Dual Branch 启动成功。97.7 samples/s，ETA 4h32m。
### [09:53] 检查点 #6

### [09:58] 检查点 #7

### [10:04] 检查点 #8

### [10:10] 检查点 #9 — 远程 ep10

**远程 Dual Branch 1-view ep10: 41.6/55.6**

| Config | 架构 | 1-view ep10 |
|--------|------|------|
| exp202 (STD-PR only) | 6-token | 43.1/56.4 |
| **exp205r (Dual Branch)** | **GCN+STD-PR** | **41.6/55.6** |
| exp203r (GCN only) | 1-pooled | 36.2/50.6 |

**Dual Branch 比 GCN-only SupCon +5.4/+5.0！** STD-PR tokens 确实在帮助 GCN。
略低于 pure STD-PR (-1.5/-0.8)——GCN 需要更多 epochs 收敛。
### [10:17] 检查点 #10

### [10:23] 检查点 #11

### [10:30] 检查点 #12

### [10:34] 检查点 #13 — 远程 ep20 ⚠️

**远程 Dual Branch 1-view ep20: 49.2/60.0**

| Config | ep10 | ep20 |
|--------|------|------|
| exp202 (STD-PR) | 43.1/56.4 | 51.8/65.3 |
| exp203r (GCN) | 36.2/50.6 | 50.3/62.7 |
| **exp205r (Dual)** | **41.6/55.6** | **49.2/60.0** |

**Dual Branch ep20 落后两个单方案！** GCN-only 50.3 > Dual 49.2。
可能双分支的梯度竞争导致两者互相拖累。
### [10:35] 本地 3-view ep10: 43.3/56.7

| Config | 3-view ep10 | 1-view ep10 |
|--------|------|------|
| exp202b (STD-PR) | 56.2/68.9 | 43.1/56.4 |
| **exp205 (Dual)** | **43.3/56.7** | **41.6/55.6** |

**3-view Dual ep10 = 1-view STD-PR ep10 (43.3 vs 43.1)！**
**远远落后 3-view STD-PR (-12.9/-12.2)！**

Dual Branch 在 3-view 下没有加速效果。GCN 的额外计算路径拖慢了 SupCon 的收敛。
可能原因：双分支的 feat list 太长 (8 items)，loss 信号被稀释。
### [10:46] 检查点 #14

### [10:52] 检查点 #15

### [10:57] 检查点 #16

### [10:59] 检查点 #17 — 远程 ep30

**远程 Dual Branch 1-view ep30: 54.8/65.0**

| Config | ep10 | ep20 | ep30 |
|--------|------|------|------|
| exp202 (STD-PR) | 43.1/56.4 | 51.8/65.3 | 57.0/69.5 |
| exp203r (GCN) | 36.2/50.6 | 50.3/62.7 | 55.3/67.9 |
| **exp205r (Dual)** | **41.6/55.6** | **49.2/60.0** | **54.8/65.0** |

**Dual 持续落后两个单方案。**
GCN 在 ep20 已反超 Dual，ep30 差距扩大。
可能原因：双分支的 CE+triplet 在 8 个 features 上分散了梯度。
### [11:06] 检查点 #18

本地 ep10=43.3/56.7 (已知)。远程 ep34。

**下一步计划** (如果 exp205 确认为负):
Swin-Small + GCN+PAA + CE + OA-SD + 3-view
OA-SD 在 CE 路线有效 (+2.9 on Tiny)。
### [11:11] 检查点 #19

远程 ep~37, 本地 ep20。远程 ep40 eval ~10min。
### [11:14] 检查点 #20

### [11:15] 检查点 #21 — 本地 ep20

**本地 3-view Dual Branch ep20: 53.1/63.2**

| Config | 3-view ep10 | 3-view ep20 |
|--------|------|------|
| exp202b (STD-PR) | 56.2/68.9 | 60.6/72.4 |
| **exp205 (Dual)** | **43.3/56.7** | **53.1/63.2** |
| delta | -12.9/-12.2 | **-7.5/-9.2** |

Gap 从 -12.9 缩小到 -7.5，但仍然巨大。
Dual Branch 3-view 在 ep20 只相当于 STD-PR 1-view ep20 (51.8/65.3)。

**初步结论**: Dual Branch 是负结果。双分支梯度竞争严重拖累收敛。
但 gap 在缩小——需要看后续是否能追上（GCN 后期通常更强）。
**决策**: 让两台继续跑完，不 kill。同时规划下一步
