# exp036 审查记录

## 第一轮审查

**审查日期**: 2026-03-13
**审查结论**: **REVIEW PASSED — ready to train**

### 逐项审查结果

| 维度 | 结论 | 备注 |
|------|------|------|
| a. design.md 设计合理性 | PASS | 单变量原则、假设清晰、预期合理 |
| b. 代码实现 | PASS | 所有6个文件逐行审查，无 bug |
| c. 配置文件 | PASS | 与 exp035a 仅差 POSE_KP_TRIPLET + OUTPUT_DIR |
| d. 数据流验证 | PASS | forward pass 追踪完整 |
| e. 梯度流 | PASS | feat_map detach 阻断 backbone，GCN params 正常接收梯度 |
| f. 向后兼容 | PASS | 默认 False 不影响已有实验 |
| g. 优化器 | PASS | 无新 learnable params |
| h. 边界情况 | PASS | 全零 keypoints 退化为均匀，mixed precision 无问题 |

### 发现的问题

| # | 严重程度 | 描述 | 影响 exp036? |
|---|---------|------|-------------|
| M1 | Medium | PDS model `_run_part_branch` 返回 kp_data 在 part_valid 中但被丢弃，kp_triplet 在 PDS 模式下静默失效 | 否（使用 PoseBackboneModel） |
| M2 | Medium | softmax sampler 的 loss_func 缺少 kp_data 参数 | 否（使用 softmax_triplet） |
| L1 | Low | design.md 列出 triplet_loss.py 但实际未修改 | 否 |
| L2 | Low | processor 中 kp_data 变量别名导致 dict 原地修改 | 否 |
