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
- [x] 启动训练
- [x] 记录前 2 个 epoch 状态

## [11:57] 早期训练状态

**状态**: ▶️ 运行中

### Epoch 1
- `Epoch 1 done`
- `Time per epoch`: `57.108s`
- `ETA`: `1h53m`
- 末段 loss:
  - `Loss`: `15.664`
  - `Acc`: `0.002`
  - `id_global`: `6.554`
  - `id_part`: `6.618`
  - `tri_global`: `7.989`
  - `tri_part`: `10.166`

### Epoch 2
- `Epoch 2 done`
- `Time per epoch`: `54.861s`
- `ETA`: `1h47m`
- 末段 loss:
  - `Loss`: `11.338`
  - `Acc`: `0.007`
  - `id_global`: `6.547`
  - `id_part`: `6.397`
  - `tri_global`: `3.433`
  - `tri_part`: `6.298`

### 当前判断
- **继续**

### 原因
1. 训练启动正常，无 `NaN / Inf / OOM`
2. `loss` 从 epoch1 到 epoch2 明显下降，符合正常 warmup 期形状
3. 当前 GPU 占用正常（约 `8GB / 24GB`），没有资源异常
