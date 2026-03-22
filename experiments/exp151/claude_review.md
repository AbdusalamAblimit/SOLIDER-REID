# exp151 PVAT Claude 广范围审查

## 审查结论：通过

### 设计审查
- 方向合理：adversarial visibility invariance 是与 PCVT 真正不同的范式
- 创新性中等：gradient reversal 本身成熟，但应用到 occluded ReID 的 visibility-invariance 场景相对新
- 定位准确：作为 supporting evidence / 机制诊断实验，不作为主创新线
- train/test 对称：✅ pvat_head 仅训练时使用

### 代码审查

#### Critical Issues: 无
#### High Issues: 无

#### Medium Issues (已解决)
1. 审查提出 warmup 期间 backbone 可能收到正向 visibility 梯度 → **分析后确认不成立**：`GradientReversal.backward` 在 `alpha=0` 时返回 `-0*grad = 0`，backbone 在 warmup 期间不收到任何 PVAT 梯度。predictor 正常学习。代码已正确。

#### Low Issues
1. Config 使用 `equal_concat`（vs exp030a base 的 `concat_scaled`）→ 与 exp030a-eq baseline 对齐，只是 test-time fusion 模式，不影响训练
2. `.to(device)` 调用冗余但无害

### 逐维度审查结果
- 默认行为安全性：✅ POSE_PVAT 默认 False
- train/test 对称：✅ pvat_head 仅在 processor 训练循环中调用
- AMP 安全：✅ GradientReversal 是标量乘法，BCE_with_logits 自动 AMP 安全
- 梯度流：✅ backbone ← -α * grad ← predictor ← BCE loss
- 优化器行为：✅ pvat_head 作为 model 子模块自动参与优化
- 单变量原则：✅ 相对 exp030a 只增加 PVAT
- 日志：✅ pvat_loss, pvat_acc, pvat_alpha, pvat_vis_ratio 可支撑止损
- warmup schedule：✅ ep1-20 alpha=0, ep21-120 线性升到 1.0
- visibility GT：✅ `pose_dict['scores'][:, 0, :]` 正确取 person-0 的 17 个分数

### 最终判断
可以启动训练。
