# exp161 STD-PR Claude 审查

## 结论：修复后通过

### High (已修复)
1. GCN+STD-PR mutual exclusion guard 缺失 → 已添加 ValueError

### Medium (不影响当前 config)
2. kp_data 缺 kp_feats/kp_weights → 当前 config 不用 MaxSim/LTCS/LPCS，安全

### Low
3. str_stats 不在训练日志中 → 后续添加

### 想法审查
- 审查认为与 exp063/081 decoder 的区别"比 design.md 声称的窄"
- 但核心区别成立：pose heatmap 作为 additive attn bias（不是 supervision/position encoding），全 768-d dim（不降维）
- 需要在论文中对 PAFormer 做清晰区分（backbone 内部 vs 后续 decoder，ViT vs Swin）

### 第一轮验证通过
- 默认行为安全 ✅
- 梯度流（detached feature map）✅
- attn_mask shape (B*num_heads, K, N) ✅
- AMP 安全 ✅
- list-loss 正确触发 ✅
- equal_concat test 正确 ✅

### 第二轮完整二次审查：PASSED
- Mutual exclusion guard 存在且正确 ✅
- attn_mask additive bias 在 MHA 中正确使用 ✅
- 完整 forward data flow 验证 ✅
- Train/test 一致性 ✅
- 所有 tensor shape/device 对齐 ✅
- Optimizer 正确 pickup 所有参数 ✅
- kp_data 安全性（无 kp_feats/kp_weights 但下游 .get() 安全）✅
- 14.2M 额外参数，在 3090 显存预算内 ✅
