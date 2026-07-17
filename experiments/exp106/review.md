# exp106 PISD 审查记录

## 审查轮次 1
| # | 严重度 | 描述 | 状态 |
|---|--------|------|------|
| 1 | CRITICAL | BN running stats 被二次 forward 污染 | ✅ 改为逐层冻结 BN |
| 2 | CRITICAL | model.eval() 导致返回格式变化 | ✅ 不用 model.eval()，逐层冻结 BN |
| 3 | MEDIUM | DropPath 二次采样导致 student 路径随机 | 接受（self-distillation 标准做法） |
| 4 | MEDIUM | dead code else 分支 | ✅ 已移除 |
| 5 | MEDIUM | GCN 在二次 forward 中不必要运行 | 接受（开销可忽略） |
| 6 | LOW | parallel_aug guard | ✅ 已加 |

## 审查轮次 2
- 验证 BN 逐层冻结实现正确
- 验证 model.training 保持 True → 返回训练格式
- ✅ 通过
