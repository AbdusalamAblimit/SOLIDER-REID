# exp044: 重建 `exp030a seed42` checkpoint — 监控日志

## 实验概述
- **目的**: 补回 `exp030a seed42` 的可复用 checkpoint 资产
- **基线配置**: `configs/occluded_duke/exp044_exp030a_seed42_rebuild.yml`
- **唯一变量**: `SOLVER.SEED = 42`
- **输出目录**: `log/occluded_duke/exp044_exp030a_seed42_rebuild`
- **说明**: 这是资产恢复，不是新方法实验

## 运行前检查
- [x] 已确认历史 `seed42` 日志存在但 checkpoint 缺失
- [x] Backbone 仍为 `Swin-Tiny`
- [x] batch size 不变
- [x] 独立 `OUTPUT_DIR`
- [ ] 启动训练
- [ ] 记录前 5 个 epoch 状态
