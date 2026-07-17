# Claude Broad Review: exp178 SupCon T=0.03 (Opus 4.6)

## 审查通过

### a. design.md
单参数消融 T=0.05→0.03。风险正确识别。

### b. 代码变更
无。CLI 覆盖 POSE_STR_SUPCON_TEMP 0.03。

### c. 数值稳定性 (T=0.03)
- max sim/T = 1.0/0.03 = 33.3
- exp(33.3) ≈ 3.6e14: fp32 安全 (max 3.4e38)
- sim_max 减法后: max exponent = 0, min exponent ≈ -66.7 → exp(-66.7) ≈ 0 (无害下溢)
- fp16 under AMP: 33.3 在 fp16 范围内 (max 65504)
- epsilon 1e-8 保护 log(0): 安全

### d. 梯度量级
- 1/T = 33.3, 是 T=0.07 的 2.3 倍
- GradScaler 提供 inf/nan 保护
- 更大梯度可能导致训练不稳定——这正是实验要测试的

### e. defaults.py
默认 0.07 不变。CLI 覆盖安全。

### f. 单变量隔离
vs exp176: 仅 temperature 0.05 → 0.03。

### g. Optimizer
SGD with cosine LR. 更大梯度 × 相同 LR = 更大参数更新。
可能导致早期不稳定但 warmup 应缓解。

零 issue。T=0.03 数值安全，梯度增大是实验目的。
