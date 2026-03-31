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

ep4/120, ETA 9h. Speed 48.3. ep10 eval ~28min。
**决策**: 继续，后台长跑
