# Claude Broad Review: exp169 PLTD (Opus 4.6 子代理审查)

## 审查通过

### 审查范围

a. **design.md**: 假设清晰，单变量原则满足（仅增加 POSE_STR_PART_DROP: 0.3）。PLTD 是零参数、零速度损耗的语义级 dropout，操作粒度（body-part token）与已有技术（neuron/pixel/region）不同。可接受的 STD-PR 扩展实验。

b. **代码审查**:
   - PLTD 实现正确（mask 生成、minimum-2-token 保证、统计记录）
   - raw_tokens/structural_tokens 别名 + 重新绑定行为验证正确
   - AMP 安全

c. **Config**: 单变量变化，与 exp166 仅差 POSE_STR_PART_DROP

d. **defaults.py**: POSE_STR_PART_DROP=0.0 默认安全，后向兼容

e. **Loss/Processor**: str_stats 含 pltd_drop，日志正确

f. **消融隔离**: 确认单变量

### 已知 Medium 问题（可接受的设计权衡）

1. **CE loss on zero tokens**: 零向量经 BN 后产生非零输出 → 分类器产生寄生梯度。但被 ~70% 正确 token 梯度稀释，BN 统计量也会适应。
2. **Triplet on zero tokens**: 零向量 L2-normalize 后仍为零 → 与所有样本等距（distance=1.0）。同样被稀释。
3. **Train-test 分布偏移**: 训练时 pooled feature 约 70% 量级，test 时 100%。BN 隐式处理（类似标准 dropout）。

这些是标准 dropout 的设计哲学（不对 dropped units 做特殊处理）。如果实验结果不佳，可在后续添加 CE/triplet masking 作为改进。
