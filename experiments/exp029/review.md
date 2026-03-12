# exp029 PSG + Pose-Weighted Pooling 审查报告

## 第一轮审查

**审查时间**: 2026-03-12
**审查代理**: Opus 4.6

### 审查结果

| 维度 | 结论 |
|------|------|
| 设计文档合理性 | PASS — 假设清晰，单变量原则满足 |
| 代码正确性（pooling 数学） | PASS — 加权平均公式正确，shape 广播正确 |
| 配置对比 | PASS — 仅 POSE_WEIGHTED_POOL 和 OUTPUT_DIR 不同 |
| 默认值安全 | PASS — False 默认值无影响 |
| 边界情况 | PASS — None heatmaps → fallback to GAP, 全低响应 → 近似 GAP |
| 梯度流 | PASS — heatmap 是外部数据(leaf tensor)，梯度仅通过 featmap |
| 后向兼容性 | PASS — 关闭时与原代码完全相同 |

### 发现的问题

| ID | 严重程度 | 描述 | 处理 |
|----|----------|------|------|
| 1 | Low | BN 可能需要适应 PWP 输出的不同幅度分布 | 监控早期训练稳定性 |
| 2 | Low | 每次 forward 的 F.interpolate 轻微开销 | 可忽略 |

### 最终结论

**审查通过，可以开始训练。**

0 Critical / 0 High / 0 Medium / 2 Low（均已记录）
