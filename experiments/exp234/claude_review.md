# exp234 Claude Review: Tiny OA-SD 240 epochs

## 审查范围

a. `design.md` — 合理性、动机、假设
b. 代码变更 — 无新代码，仅 config override
c. 默认值安全性 — 无新 config key
d. 与 exp191 对照 — 单变量隔离
e. LR schedule / 训练动态 — 是否因 epoch 变化产生副作用

---

## 1. design.md 审查

**动机合理**。exp233 显示 per-part id loss 在 ep70 才降到 3.0（baseline ep30 已 <1.0），提出 120 epoch 可能是次优训练长度，这是一个值得验证的假设。延长训练是廉价的对照实验，不需要代码修改。

**核心假设清晰**：如果 baseline 在 240ep 显著优于 120ep，说明 120ep 次优。反过来，如果 240ep 不优于 120ep，说明 120ep 已收敛，后续创新实验无需延长。

**早停规则合理**：ep120 < 63.2% 视为异常。这是一个必要的 sanity check。

**单变量原则**：仅改 MAX_EPOCHS (120→240) 和 CHECKPOINT_PERIOD (相应调整为 40)，完全满足单变量要求。

**关于"是否只是小调参"的质疑**：这确实是一个只改配置参数的实验，但它的目的是诊断性的——确认当前训练长度是否是瓶颈。这不是在某个方向上做微调逃避创新，而是为后续所有实验建立正确的训练基线。如果 240ep 确实显著更好，那之前所有 120ep 的创新实验都需要重新解读。这类诊断实验是合理的。

## 2. 代码变更审查

无代码变更。Command-line override:
- `SOLVER.MAX_EPOCHS 240`
- `SOLVER.CHECKPOINT_PERIOD 40`
- `OUTPUT_DIR ./log/occluded_duke/exp234_oasd_240ep`

所有其余配置与 exp191 完全相同（swin_tiny.yml base + OA-SD overrides）。

## 3. LR Schedule 交互分析

**重要发现**：项目使用 `WARMUP_METHOD: 'cosine'`，对应 `CosineLRScheduler`，其 `t_initial=num_epochs=MAX_EPOCHS`。

这意味着：
- 120ep 时：LR 在 ep120 衰减至 lr_min (0.002 * BASE_LR)
- 240ep 时：LR 在 ep240 才衰减至 lr_min，衰减速度减半

这不是简单地"在 ep120 之后继续跑相同的学习率"，而是改变了整个 LR schedule 的形状。ep120 时 240ep 实验的 LR 大约在 lr_min 附近但尚未完全衰减，而 120ep 实验此时已到最低。

**这是否构成"非单变量"？** 严格来说，cosine schedule 自动适应 MAX_EPOCHS 是预期行为。如果固定 120ep 的 schedule 然后在之后用最低 LR 继续训练，那才是"纯延长"。当前做法是"用 240ep 的 schedule 训练 240ep"，这是更合理的做法，因为 cosine schedule 需要知道总 epoch 数。

**建议关注**：在 monitor.md 中记录 ep60 和 ep120 时的 LR 值，与 exp191 同时刻对比，以理解 LR schedule 差异的影响。

## 4. EMA Teacher 兼容性

OA-SD 的 EMA teacher (decay=0.999) 是逐步更新的，与 epoch 数无关，不会因延长训练产生问题。更长的训练意味着 teacher 有更多更新步，理论上更精确。

## 5. CHECKPOINT_PERIOD=40 审查

240/40=6 个 checkpoint，合理。ep40, ep80, ep120, ep160, ep200, ep240。ep120 checkpoint 可直接与 exp191 final 对比。

## 6. 默认值安全性

无新 config key 引入。所有 override 都是已有参数。不影响其他实验的可复现性。

## 7. 风险评估

- **低风险**：无代码变更，最坏情况是浪费 2x 训练时间
- **注意事项**：如上所述，LR schedule 改变意味着 ep120 结果不能直接与 exp191 对比（因为 LR 不同）。真正的对比是 exp234 best epoch vs exp191 best epoch (ep120)

## 8. 论文价值判断

作为 supporting evidence 合理：如果 240ep 不优于 120ep，确认训练长度不是瓶颈，给后续创新实验一个安心的基准。如果 240ep 显著更好，则所有后续实验应调整为更长训练。

这不是主创新方向，是必要的工程诊断。

---

## 审查结论

无 Critical / High / Medium / Low 级别问题。

- 实验设计合理，单变量隔离良好
- 无代码变更，无引入 bug 的风险
- LR schedule 自适应是预期行为，但需在监控中注意
- 建议在 monitor.md 中记录 LR 曲线差异

**审查通过**
