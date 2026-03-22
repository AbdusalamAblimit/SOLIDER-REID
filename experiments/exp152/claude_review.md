# exp152 MaxSim Triplet Claude 广范围审查

## 审查结论：通过（修复后）

### Critical Issues (已修复)
1. **C1: margin=None TypeError** — NO_MARGIN=True 时 triplet.margin=None, 导致 F.relu(tensor + None) crash → 已修复：添加 soft margin (F.softplus) 分支

### High Issues (已修复)
1. **H1: 零 loss 的断开梯度图** — new_zeros(requires_grad=True) 创建 leaf tensor → 已修复：改用 (kp_feats * 0.0).sum()

### Medium Issues
1. **M1: 非对称距离矩阵** — 设计如此（ColBERT 范式），不需修改
2. **M2: AMP fp16 精度** — cos/0.05 最大值=20, exp(20)=4.8e8, 在 float32 安全。需要监控训练稳定性
3. **M3: 注意力熵日志** → 已添加 `maxsim_ent`

### Low Issues
1. **L2: 设计文档与实际 margin 不一致** — 实际用 soft margin, design.md 写 hard margin. 不影响运行
2. **L3: POSE_TEST_FEAT=equal_concat** — 与 exp030a-eq baseline 对齐，单变量成立

### 验证通过的维度
- 默认行为安全 ✅
- 梯度流正确 ✅
- 内存 ~10MB overhead ✅
- 数值稳定（softmax max-subtraction 处理）✅
- Hard mining 自排除正确 ✅
- 单变量原则 ✅

### 最终判断
修复 C1/H1/M3 后可以启动训练。
