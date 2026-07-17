Now I have all the information needed for a thorough review.

## exp133 代码接线审查

### 1. 审查结论：允许启动

### 2. HIGH（阻塞项）

无。

### 3. MEDIUM（非阻塞项）

- **训练/测试 base_dist 计算方式硬编码 vs 配置**：训练端 `_compute_lpcs_loss` 硬编码 `base_dist = 0.5 * (global_dist + kp_dist)`，评估端 `metrics.py:286` 使用 `(gw * global_dist + kw * kp_dist) / (gw + kw)`，当前默认 gw=kw=1.0 两者结果一致。但如果未来修改 `CVK_GLOBAL_WEIGHT / CVK_KP_WEIGHT`，会导致 train-test 不一致。当前实验不受影响，但建议后续统一为配置驱动。

- **warmup 期间 weight decay 空耗 LPCS head 参数**：前 20 epoch LPCS head 无梯度信号，但 optimizer 的 weight decay（1e-4）持续作用于其 Xavier-initialized 隐藏层权重。20 epoch 后权重衰减幅度极小（SGD + wd=1e-4），且与 LTCS 采用相同模式，exp132 已验证可接受。

### 4. LOW（可后续优化）

- **ranking loss 中 Python for-loop**：`_compute_lpcs_loss` 对 batch_size=64 做 Python-level 循环构建 pairwise ranking loss。每次迭代的张量很小（~3 positives × ~60 negatives），不会成为瓶颈，但如果未来扩大 batch size 可考虑向量化。

- **teacher 信号退化为零的边界**：若 teacher_dist 与 base_dist 完全相同（极端情况），`pair_change = 0 → pair_weight = 0 → loss = 0`，head 无学习信号。实际上 20 epoch warmup 后 bank 已充分积累，teacher 与 base 必有差异，此边界在实践中不会触发。

### 5. 结论

| 检查项 | 结果 |
|--------|------|
| 单变量性 | ✅ 对照 exp132：LTCS 关闭（config 未设 POSE_LTCS，默认 False），LPCS 开启（`POSE_LPCS: True`），互斥校验正确。其余配置（PSG、GCN、训练超参）与 exp132 一致。唯一变量为 alpha-fusion → pair residual scorer |
| checkpoint 接线 | ✅ `lpcs_head` 是 `PoseBackboneModel` 的 `nn.Module` 子模块（`self.lpcs_head`），自动进入 `state_dict()` → 保存/加载正确。`make_optimizer` 遍历 `model.named_parameters()` → LPCS head 参数自动纳入优化器 |
| train loss 接线 | ✅ `_compute_lpcs_loss` 在 warmup 后被调用（`epoch > 20`），通过 `lpcs_head(desc)` 产出 delta → softplus ranking loss → 加权回传。所有非-head 输入均 `.detach()`，梯度仅流经 MLP 权重，不干扰 backbone。loss 正确累加到主 loss（`loss = loss + lpcs_weight * lpcs_loss`），在 `scaler.scale(loss).backward()` 之前完成 |
| test/evaluator 接线 | ✅ `evaluator.pair_residual_head` 在 validation（周期 eval + 最终 eval）和 `do_inference` 中均已正确赋值为 `_eval_model.lpcs_head`。`metrics.py` 的 `cvk_residual` 分支：构建 descriptor → head 预测 delta → `base_dist + delta` → distmat，chunked（256）处理避免 OOM，device 转换正确 |
| ranking loss 风险 | ✅ 正负方向正确：`softplus(d_pos - d_neg)` 在 d_pos > d_neg 时产生高 loss，推动正对距离下降、负对距离上升。标签掩码：`same_label & ~eye` / `~same_label` 无误。teacher weight 归一化有 `clamp(min=1e-6)`。OOM 安全：每 query 的 rank_term 矩阵 ≈ (3, 60)，总量极小。`delta_scale=0.5` + tanh bound 保证修正幅度可控 |
