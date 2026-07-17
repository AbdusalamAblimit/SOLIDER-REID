# exp051 PAML 代码审查记录

## 第一轮审查

**审查范围**: design.md, defaults.py, make_loss.py, processor.py, config yml

### 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | LOW | pose_psg_gcn_paml.yml | POSE_TEST_FEAT 从 concat_scaled 改为 equal_concat，但这只影响评估不影响训练，且 design.md 已说明对照的是 exp030a equal_concat 模式 | 接受 |
| 2 | LOW | make_loss.py | POSE_KP_TRIPLET 和 POSE_PAML 同时启用时会叠加，但当前配置不会出现 | 接受 |
| 3 | LOW | make_loss.py | `y = dist_an.new_ones()` 与 TripletLoss 中的风格略有不同 | 接受（功能正确） |

### 审查通过项

- ✅ _compute_paml_triplet 逐关键点距离计算正确
- ✅ confidence 加权使用 min(score_i, score_j) 正确匹配 CVK 逻辑
- ✅ hard_example_mining 正确导入和调用
- ✅ POSE_PAML 默认为 False，不影响已有实验
- ✅ kp_data 在 PAML 启用时正确传递
- ✅ 梯度流正确：通过 GCN 参数，不通过 backbone（detached）
- ✅ NaN/Inf 风险充分处理（clamp）
- ✅ 当 POSE_PAML=False 时训练路径与之前完全一致
- ✅ 内存开销极小（< 5MB）
- ✅ amp.autocast 兼容

### 结论

✅ **通过** — 无 Critical 或 High 级别问题。实现正确、干净、隔离良好。可以开始训练。
