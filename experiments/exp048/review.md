# exp048 SGMKC 代码审查报告

## 审查轮次: Round 1 (2026-03-13)

### 审查结果: PASS

### 逐项审查

| 维度 | 结论 | 说明 |
|------|------|------|
| design.md | PASS | 假设清晰，单变量原则，预期结果合理 |
| config (exp048 vs exp030a) | PASS | 除 SGMKC 3 个参数外完全一致 |
| defaults.py | PASS | 默认值正确，不影响已有实验 |
| skeleton_gcn.py | PASS | masking 位置正确（GCN 前），detach+clone 正确 |
| pose_backbone_model.py | PASS | 参数传递正确 |
| processor.py | PASS | 重建 loss 计算正确，mask 反转正确 |
| 梯度流 | PASS | SGMKC loss 只训练 GCN，不影响 backbone |
| 设备一致性 | PASS | 所有 tensor 在同一设备 |
| dtype/AMP | PASS | float16 下 MSE loss 数值稳定 |
| 边界情况 | PASS | 全 mask/无 mask 场景均正确处理 |

### 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|----------|------|------|------|
| 1 | Low | skeleton_gcn.py | 步骤编号重复（两个 #3） | 已修复 |
| 2 | Medium | processor.py | AMP autocast 下 MSE loss — 理论安全但需监控 | 不需修改 |

### 最终结论
审查通过，可以开始训练。
