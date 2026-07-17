# exp028 PDS + Part LR Boost 审查报告

## 第一轮审查

**审查时间**: 2026-03-11
**审查代理**: Opus 4.6

### 审查结果

| 维度 | 结论 |
|------|------|
| 设计文档合理性 | PASS — 假设清晰，单变量原则满足 |
| 代码正确性 | PASS — LR 因子正确应用于 Part 参数 |
| 配置对比 | PASS — 仅 POSE_PART_LR_FACTOR 和 OUTPUT_DIR 不同 |
| 默认值安全 | PASS — 1.0 默认值无影响 |
| Warmup 调度器交互 | PASS — 每组 base_values 正确保留 LR 比例 |
| SGD Momentum | PASS — per-param-group LR 与 momentum 兼容 |
| 梯度流 | PASS — stop_grad detach 隔离 Part 梯度，LR boost 不泄漏 |

### 发现的问题

| ID | 严重程度 | 描述 | 处理 |
|----|----------|------|------|
| 1 | Medium | Part bias 参数受 BIAS_LR_FACTOR(2x) * PART_LR_FACTOR(3x) = 6x 叠加 | 已在 design.md 中注明，不修改（合理的隐含行为） |
| 2 | Low | design.md 提到 `part_heads` 但代码匹配 `part_stage3/norm3/pooling` | 已修复 |
| 3 | Low | design.md 提到修改 model 文件但实际未修改 | 已修复 |
| 4 | Low | 日志只显示第一个 param group 的 LR，Part 实际 LR 不可见 | 在 monitor.md 中注明 |

### 最终结论

**审查通过，可以开始训练。**

0 Critical / 0 High / 1 Medium（已记录） / 3 Low（2个已修复，1个已记录）
