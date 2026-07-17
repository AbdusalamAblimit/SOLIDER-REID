# exp150 PCVT-Random Claude 广范围审查

## 审查结论：通过

### 设计审查
- 设计合理，是 exp148 PCVT 的核心机制对照
- 满足 CLAUDE.md "同一主问题下两个能回答关键机制问题的强对照" 要求
- 单变量原则严格：唯一差异为 masking 策略（pose-guided → random block）
- 预期结果分析充分（3 种场景均有明确解读）

### 代码审查

#### Critical Issues: 无
#### High Issues: 无
#### Medium Issues: 无

#### Low Issues (已修复)
1. **L1**: design.md 中图像尺寸写成 256×128，实际为 384×128，块尺寸为 48×32 而非 32×32 → **已修正**
2. **L2**: GLOBAL_LOSS_SCALE 确认两个 config 均通过 GCN list-loss path 自动提供 0.5x → **非问题**

### 逐维度审查结果
- 默认行为安全性：✅ PCVT_RANDOM 默认 False，不影响任何现有实验
- train/test 对称：✅ Random masking 只在训练时生效
- AMP 安全：✅ 所有新 tensor 类型与 AMP 兼容
- 数据流完整性：✅ meta dict 9 个键与 pose-guided 模式一致
- block 尺寸计算：✅ 384/8=48, 128/4=32, 完美整除
- 单变量原则：✅ config 唯一差异为 PCVT_RANDOM=True 和 OUTPUT_DIR
- 日志一致性：✅ pcvt_* 日志键完整，后续可直接与 exp148 对比

### 最终判断
可以启动训练。
