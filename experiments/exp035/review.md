# exp035 审查记录

## 第一轮审查

**审查日期**: 2026-03-13
**审查结论**: REVIEW FAILED

### 逐项审查结果

| 维度 | 结论 | 备注 |
|------|------|------|
| a. design.md 设计合理性 | PASS | 单变量原则、baseline 正确 |
| b. 代码实现 | PASS (Medium issues) | _compute_kp_weights 正确，kp_scores 返回值未使用 |
| c. 配置文件 | PASS | 四个 config 仅差 POSE_KP_WEIGHT_MODE 和 OUTPUT_DIR |
| d. defaults.py | PASS | 默认值 'score' 不影响已有实验 |
| e. Forward pass 数据流 | FAIL | 数据管线 visibility 未在增强中正确处理 |
| f. 梯度流 | PASS | 无新 learnable params |
| g. 优化器 | PASS | 无新参数 |
| h. 向后兼容 | PASS | 默认 'score' = 旧行为 |

### 发现的问题

| # | 严重程度 | 描述 | 文件 |
|---|---------|------|------|
| H1.1 | **High** | `_joint_flip` 未对 visibility/visibility_binary 做 L-R swap | pose_dataset.py:350 |
| H1.2 | High | `_joint_pad_crop` 未对 OOB 关键点清零 visibility | pose_dataset.py:388 |
| H1.3 | High | Random erasing 未对被擦除关键点清零 visibility | pose_dataset.py:156 |
| M1 | Medium | kp_scores 返回值不再使用 | skeleton_gcn.py:198 |
| M2 | Medium | binary_visibility 全零时退化为均匀平均 | skeleton_gcn.py:254 |

## 第二轮审查

**审查日期**: 2026-03-13
**审查结论**: **REVIEW PASSED — ready to train**

所有三个 High issues 已修复：
- H1.1: 在 `_joint_flip` 的 FLIP_PAIRS 循环中添加 visibility/visibility_binary swap
- H1.2: 在 `_joint_pad_crop` 的 OOB 处理中添加 visibility/visibility_binary 清零
- H1.3: 在 random erasing 中添加 visibility/visibility_binary 清零

M1、M2 为非阻塞性问题，不影响训练正确性。
