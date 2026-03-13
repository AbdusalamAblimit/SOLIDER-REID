# exp037 审查记录

## 编号说明
- `exp037` 编号已偏离最初的 visibility 路线命名。
- 本审查对应的是 `LKA` 这一 GCN branch 内部改动，不应被视为 visibility 主线实验审查。

## 第一轮审查

**审查日期**: 2026-03-13
**审查结论**: **REVIEW PASSED — ready to train**

### 逐项审查结果

| 维度 | 结论 | 备注 |
|------|------|------|
| a. design.md 设计合理性 | PASS | 单变量原则、假设清晰 |
| b. 代码实现 | PASS | LKA MLP 构造、zero-init、forward 使用全部正确 |
| c. 配置传递 | PASS | pose_backbone_model + pose_dual_stream_model 均正确传参 |
| d. 配置文件 | PASS | 与 exp035a 仅差 POSE_KP_LEARNABLE_ATTN + OUTPUT_DIR |
| e. defaults.py | PASS | 默认 False 不影响已有实验 |
| f. 梯度流 | PASS | LKA 通过 kp_weights→skeleton_feat→loss 路径正常接收梯度 |
| g. 优化器 | PASS | ~600 新参数自动注册到优化器 |
| h. 向后兼容 | PASS | 默认 False 时完全等价 |

### 发现的问题
无
