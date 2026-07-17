# exp031 多种子验证 — 汇总日志

## 实验配置
- **日志目录**: `4090_log/multiseed/`
- **种子**: `1234`, `42`, `2024`
- **已完成配置**:
  - `exp000` Baseline
  - `exp007` PSG
  - `exp007a` PSG + 0.5x global loss
  - `exp023` PDS+StopGrad（global-only）
  - `exp030a` PSG + Skeleton GCN（`global` / `concat_scaled` / `equal_concat` / `gcn_only`）
- **统计口径**: 统一使用 **two-sided paired t-test on mAP**

---

## 结果总表

| 方法 | 模式 | Seed 1234 | Seed 42 | Seed 2024 | Mean±Std (mAP) | Mean±Std (R1) |
|------|------|-----------|---------|-----------|----------------|---------------|
| Baseline (exp000) | global | 56.7% | 55.9% | 56.9% | **56.50±0.53%** | **66.33±0.67%** |
| PSG (exp007) | global | 58.3% | 57.9% | 57.3% | **57.83±0.50%** | **67.13±0.84%** |
| PSG + 0.5x loss (exp007a) | global | 59.6% | 59.5% | 59.0% | **59.37±0.32%** | **69.43±0.12%** |
| PDS+StopGrad (exp023) | global | 59.7% | 59.2% | 58.7% | **59.20±0.50%** | **68.63±0.47%** |
| PSG + GCN (exp030a) | global | 59.8% | 59.1% | 59.1% | **59.33±0.40%** | **68.87±1.00%** |
| PSG + GCN (exp030a) | concat_scaled | 60.5% | 59.7% | 60.4% | **60.20±0.44%** | **73.13±0.29%** |
| PSG + GCN (exp030a) | equal_concat | 61.1% | 60.2% | 60.9% | **60.73±0.47%** | **72.57±0.58%** |
| PSG + GCN (exp030a) | gcn_only | 58.2% | 57.4% | 58.3% | **57.97±0.49%** | **71.77±0.60%** |

---

## 关键统计检验

| 对比 | Mean Δ | Paired Diffs | t-stat | p-value | 结论 |
|------|--------|--------------|--------|---------|------|
| PSG vs Baseline | **+1.33%** | (1.6, 2.0, 0.4) | 2.77 | 0.1091 | 3 个 seed 全正，但 n=3 时双侧检验仍偏弱 |
| exp007a vs PSG | **+1.53%** | (1.3, 1.6, 1.7) | 12.76 | 0.0061 | ✅ 0.5x global loss 是稳定增益 |
| exp007a vs exp023-g | **+0.17%** | (-0.1, 0.3, 0.3) | 1.25 | 0.3377 | 无显著差异；exp023 global 基本被 exp007a 复现 |
| exp030a-g vs exp007a | **-0.03%** | (0.2, -0.4, 0.1) | -0.18 | 0.8740 | GCN 分支对 global 基本中性 |
| exp030a-cs vs exp030a-g | **+0.87%** | (0.7, 0.6, 1.3) | 3.96 | 0.0581 | 边缘改善，方向一致 |
| exp030a-eq vs exp030a-g | **+1.40%** | (1.3, 1.1, 1.8) | 6.73 | 0.0214 | ✅ fusion 增益成立 |
| exp030a-eq vs exp030a-cs | **+0.53%** | (0.6, 0.5, 0.5) | 16.00 | 0.0039 | ✅ `equal_concat` 明显优于 `concat_scaled` |
| exp030a-eq vs exp007a | **+1.37%** | (1.5, 0.7, 1.9) | 3.87 | 0.0606 | 边缘改善，说明 branch 增益大体稳定 |
| exp030a-eq vs exp023-g | **+1.53%** | (1.4, 1.0, 2.2) | 4.35 | 0.0491 | ✅ 当前最强 fusion 已稳定超过 PDS global |

---

## 修正后的结论

1. **PSG 依然成立，但表述要收敛**
   3 个 seed 全部优于 baseline，均值 `56.50% -> 57.83%`。这足够说明 PSG 是稳定正向的，但在 `n=3`、双侧检验下不应再写成“统计显著已完全确认”。

2. **`0.5x global loss` 不是训练方差假象**
   `exp007a` 相对 `exp007` 的 3-seed 改善为 `+1.53% mAP`，而且三个 paired diffs 都在 `+1.3% ~ +1.7%`。此前“0.5x 只是异常高 seed”的判断已经被推翻。

3. **PDS+StopGrad 的 global 增益基本可由 `0.5x loss` 解释**
   `exp007a = 59.37%`，`exp023-g = 59.20%`，差异仅 `+0.17%` 且无显著性。更准确的表述是：PDS/StopGrad 在 global-only 指标上的收益，主要来自它隐式带来的 loss weighting，而不是双流架构本身。

4. **GCN/KPP branch 的正确读法是“fusion 增益”，不是“global 增益”**
   `exp030a-global = 59.33%` 与 `exp007a = 59.37%` 几乎相同；真正的提升发生在 fusion：`exp030a-eq = 60.73%`，对自身 global 稳定 `+1.40%`。

5. **`equal_concat` 应替代 `concat_scaled` 成为主测试模式**
   三个 seed 中 `equal_concat` 都优于 `concat_scaled`，均值差 `+0.53%`，而且非常一致 `(0.6, 0.5, 0.5)`。此前把 `concat_scaled` 写成主模式的结论需要撤回。

6. **exp030b 和 exp032 的角色要重新定义**
   - `exp030b` 只能说明：`w_p=0.01` 时 branch 几乎未训练好，因此 fusion 不稳定。
   - `exp032` 说明：keypoint pooling 本身就是强基线。
   - 两者合起来支持的应是“branch 强度主要先来自 sparse keypoint pooling，GCN 负责 refinement”，而不是“GCN 完全无效”或“所有单 seed 差异都只是方差”。

---

## 当前最可靠的论文级证据链

1. `Baseline -> PSG`: `56.50% -> 57.83%`
   说明 backbone 内部 pose gate 稳定有效。

2. `PSG -> PSG + 0.5x loss`: `57.83% -> 59.37%`
   说明更弱的 global 梯度是一个真实的训练配方增益。

3. `PSG + 0.5x loss -> PSG + GCN(global)`: `59.37% -> 59.33%`
   说明 branch 训练本身不改变 global。

4. `PSG + GCN(global) -> PSG + GCN(equal_concat)`: `59.33% -> 60.73%`
   说明训练好的 branch 在检索时提供了稳定互补信息。

当前最稳的无后处理主结果应写为：
**`PSG + 0.5x global loss + skeleton branch (equal_concat)` = `60.73±0.47% mAP`, `72.57±0.58% R1`**
