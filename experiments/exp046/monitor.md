# exp046: 重建 `exp030a seed2024` checkpoint — 监控日志

## 实验目标
- **目的**: 重建 `exp030a seed2024` checkpoint，补齐第三个可复用资产
- **配置**: `configs/occluded_duke/exp046_exp030a_seed2024_rebuild.yml`
- **输出目录**: `log/occluded_duke/exp046_exp030a_seed2024_rebuild`

## 启动前检查
- [x] Backbone 固定为 `Swin-Tiny`
- [x] batch size 保持 `64`
- [x] 相对 `exp030a` 只改 `SOLVER.SEED` 与 `OUTPUT_DIR`
- [x] 默认 `POSE_TEST_FEAT` 保持 `concat_scaled`
- [x] 训练结束后再补 `equal_concat / cvk_hybrid`

---
## [启动] 训练计划

### 启动命令
- `/root/miniconda3/envs/solider-reid/bin/python -u train.py --config_file configs/occluded_duke/exp046_exp030a_seed2024_rebuild.yml`

### 监控节奏
- Epoch 1-5：约每 2 分钟检查一次
- Epoch 6-30：约每 3 分钟检查一次
- Epoch 30+：约每 5 分钟检查一次

### 当前判断
- **继续**

### 原因
1. `exp045` 已把第二个 checkpoint 证据补齐，下一步最自然的高价值动作就是补第三个 seed 资产。
2. 在资产未补齐前，继续做新的 CVK 调参收益很低。
