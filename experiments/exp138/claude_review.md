I've completed a thorough review of all files. Here is my assessment:

---

# Claude 审查结论

## 结论
- **允许启动**

## Blocking
- 无

## Major
- 无

## Medium

**M1: tau=8 对 pos/neg 的衰减效果高度不对称（非 bug，但影响实验解释）**

在典型 batch（64 images, 4 instances/class）中：
- **Positive 数量 ≈ 3**：rank factors = [1.0, 0.88, 0.78]，几乎是均匀加权
- **Negative 数量 ≈ 60**：rank factors = [1.0, 0.88, ..., exp(-59/8)≈0.0006]，极其激进

这意味着 `rank_decay` 在实践中主要是**单侧 soft hard-negative mining**，而对 positive 侧几乎没有影响。这不是实现 bug（代码完全正确），但与 design.md 中"对 positive 和 negative 都做平滑 rank 强调"的叙述有偏差。

**影响**：不阻塞启动，但如果实验结果不理想，应考虑 pos/neg 分别设 tau（如 `tau_pos=2, tau_neg=8`），或仅对 negative 做 rank decay。

**M2: `lpcs_rwm` 指标混合了 pos/neg rank factors，掩盖不对称性**

`rank_weight_mean` 在 L493-494 / L516 处将 pos 和 neg 的 rank factors 合并为单一均值。由于 neg 数量远大于 pos（~60 vs ~3），该指标几乎完全由 neg 侧决定。如果 pos 侧退化为均匀但 neg 侧衰减正常，`lpcs_rwm` 仍然会显示"衰减有效"。

**建议**：如果需要精细诊断，可在后续版本分别记录 `lpcs_rwm_pos` / `lpcs_rwm_neg`。当前不阻塞。

## Low

**L1: `_rank_decay_factors` 在 AMP autocast 下接收 float16 输入**

`_compute_lpcs_loss` 在 `amp.autocast()` 内被调用。`final_dist` 可能是 float16，传入 `_rank_decay_factors`。对于当前 batch size（≤64 items per class），rank 值很小（0~60），`exp(-60/8)≈0.0006` 在 float16 精度范围内（min subnormal ≈ 6e-8）。**不会导致数值错误**，但如果未来 batch size 增大或 tau 减小，可能需要 `.float()` 转换。

**L2: `POSE_LPCS_RANK_TAU: 8.0` 在 config 中显式设置但等于默认值**

这不是错误（反而增加了可读性），但 exp135 config 依赖默认值 `8.0` 而未显式设置。两者行为一致，但风格略有不同。

## 建议

1. **可以启动实验**。代码实现正确，单变量原则满足，默认行为未被破坏，监控指标足够验证机制是否激活。
2. 启动后关注 `lpcs_rwm` 是否显著 < 1.0（预期 ≈ 0.3-0.5，确认 rank decay 非退化），`lpcs_rsr` 是否 = 1.0（确认未丢弃 pairs）。
3. 如果 R1 未见改善，M1 指出的 pos/neg 不对称可能是关键线索——当前 positive 侧的 rank decay 几乎无效，改进空间在 negative 侧。
4. M2 的分离日志建议可以在后续实验中追加，不影响当前启动。
