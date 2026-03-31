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
**决策**: 继续观察到 ep30-40，如果仍大幅落后则可能是负结果
