# exp216 审查报告

## 审查范围

a. design.md 合理性
b. 调度器/训练代码兼容性
c. 配置变更隔离性
d. 与前序实验对照

## a. design.md 审查

**实验**: exp216 — Small GCN+PAA+CE+OA-SD, 200 epochs (vs exp206r 120 epochs)

设计文档清晰，单变量原则满足（仅 MAX_EPOCHS 120→200）。

**关于"是否只是小调参"的质疑**:
这是一个纯超参数实验，改动量极小。但作为训练长度扫描（120→200），它的合理性在于：
- exp206r 学习曲线显示 ep100-120 仍在增长 (+0.1-0.3%/10ep)
- 如果 200ep 显著提升，说明之前所有 Small 实验都在欠训练
- 这为后续实验设定正确的训练长度基准
- 负面结果（无提升）也有价值：确认 120ep 已足够收敛

**判定**: 合理的基准实验。不作为创新主线，但对校准后续实验有价值。

## b. 调度器分析

### CosineLRScheduler 参数 (scheduler_factory.py)

```
t_initial = MAX_EPOCHS (200)
warmup_t = WARMUP_EPOCHS (20)
warmup_prefix = False (default)
t_mul = 1.0
cycle_limit = 1
lr_min = 0.002 * BASE_LR
warmup_lr_init = 0.01 * BASE_LR
```

### 200 epochs 下的调度行为

**Warmup (epoch 0-19)**: 线性从 `0.01*BASE_LR` 升到 `BASE_LR`。与 120ep 完全相同，无问题。

**Cosine decay (epoch 20-199)**: 由于 `warmup_prefix=False`，cosine 周期覆盖整个 `t_initial=200` 范围。

- Warmup→Cosine 过渡点 (t=20):
  - 120ep: `cos(π*20/120) = cos(π/6) ≈ 0.866` → LR ≈ 93.3% of peak
  - **200ep: `cos(π*20/200) = cos(π/10) ≈ 0.951` → LR ≈ 97.6% of peak**
  - 200ep 的过渡更平滑（更接近 peak），这是正向的

- Cosine 尾部衰减:
  - 120ep: 在 ep100-120 LR 快速下降到 lr_min
  - 200ep: 在 ep100-120 LR 仍在 ~18-30% of peak，ep160-200 才进入快速下降
  - **这意味着 ep100-120 区间的学习率显著高于 120ep 版本**，模型继续积极学习

- 最终 LR:
  - 两者都衰减到 `lr_min = 0.002 * BASE_LR`
  - 120ep 在 ep119 达到 lr_min
  - 200ep 在 ep199 达到 lr_min

**结论**: 调度器正确处理 200 epochs。无 bug，无边界问题。

### Warmup 比例变化

- 120ep: 20/120 = 16.7% warmup
- 200ep: 20/200 = 10.0% warmup

历史参考: exp011 (Tiny PSG 200ep) 使用了 WARMUP_EPOCHS=30 (15%)。exp216 保持 WARMUP_EPOCHS=20 (10%)。

**10% warmup 是否太短？** 不太可能有问题：
1. 20 epoch 的绝对 warmup 长度与 120ep 版本相同
2. Swin-Small 已用 SOLIDER 预训练权重初始化，不需要更长的 warmup
3. exp011 (Tiny, 30ep warmup) 并未优于 exp007 (120ep, 20ep warmup)，说明额外 warmup 无价值
4. warmup 的目的是避免初始大 LR 的不稳定，20 epoch 已足够

**无需修改 WARMUP_EPOCHS**。

## c. 配置隔离性

exp216 仅修改 `SOLVER.MAX_EPOCHS: 200`（通过命令行 override 或独立 config）。

**CHECKPOINT_PERIOD 注意事项**: 
- exp206r 的 config 中 CHECKPOINT_PERIOD=120，仅保存最终模型
- 200ep 运行应设置 `CHECKPOINT_PERIOD=20`，以便：
  1. 对比 ep120 checkpoint（与 exp206r 直接对比）
  2. 监控 ep120 后的增益曲线
  3. 早停止损

**建议**: 启动命令中加 `SOLVER.CHECKPOINT_PERIOD 20`。

**EVAL_PERIOD**: 保持 10 即可，200ep 每 10ep 评估一次，共 20 次评估。

## d. 与前序实验对照

### 历史 200ep 经验

| 实验 | Backbone | 120ep mAP | 200ep mAP | delta |
|------|----------|-----------|-----------|-------|
| exp007 vs exp011 | Tiny PSG | 58.3% | 58.3% | 0.0% |

exp011 (Tiny 200ep) 没有任何提升。但 Tiny 和 Small 有差异：
- Small 容量更大 (50M vs 28M params)，可能需要更多 epoch
- exp206r 学习曲线 ep100-120 仍在增长，而 exp007 在 ep100 已饱和

### 风险评估

- **成功概率**: ~30%。历史 200ep 无提升 (exp011)，但 Small 可能不同。
- **预期增益**: +0.5-1.0% mAP（如果有效）
- **风险**: 浪费 ~4 小时 GPU 时间（200ep Small ≈ 8h, 其中前 120ep 与 exp206r 重复）
- **最坏情况**: 与 exp206r 一样或更差（overfitting on small dataset）

## e. 潜在问题检查

1. **OA-SD EMA teacher**: EMA 更新与 epoch 数无关（per-step），200ep 只是更多步。无问题。
2. **Label smoothing**: off。无问题。
3. **ROA/PLBOA 数据增强**: 与 epoch 数无关。无问题。
4. **Swin DropPath**: 固定概率，与 epoch 数无关。无问题。
5. **Weight decay**: 固定值 1e-4，与 epoch 数无关。无问题。

## 审查结论

**无 Critical / High / Medium 问题。**

**Low**:
1. 确保 CHECKPOINT_PERIOD=20（不要用默认 120），以获得中间 checkpoint 用于分析。
2. 确保 EVAL_PERIOD=10 以监控后期增长趋势。

两项均为启动命令层面的建议，不影响代码正确性。

## 审查通过

exp216 是一个清晰的单变量超参数实验。cosine 调度器正确处理 200 epochs，warmup 行为合理，无代码/配置兼容性问题。可以启动训练。
