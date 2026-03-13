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

---
## [12:06] 检查点 #2

**状态**: ▶️ 运行中

### 训练进度
- 已完成：`Epoch 10/120`
- 当前：`Epoch 11` 已开始
- 近期 epoch 耗时：约 `54.5s ~ 55.3s`
- 最新 ETA：约 `1h41m`

### 训练 loss 变化
- `Epoch 3` 末段：
  - `Loss`: `9.141`
  - `Acc`: `0.064`
  - `id_global`: `6.523`
  - `id_part`: `6.082`
  - `tri_global`: `1.775`
  - `tri_part`: `3.902`
- `Epoch 5` 末段：
  - `Loss`: `7.724`
  - `Acc`: `0.222`
  - `id_global`: `6.413`
  - `id_part`: `5.469`
  - `tri_global`: `0.992`
  - `tri_part`: `2.574`
- `Epoch 8` 末段：
  - `Loss`: `6.659`
  - `Acc`: `0.233`
  - `id_global`: `6.052`
  - `id_part`: `4.699`
  - `tri_global`: `0.635`
  - `tri_part`: `1.931`
- `Epoch 10` 末段：
  - `Loss`: `6.154`
  - `Acc`: `0.227`
  - `id_global`: `5.683`
  - `id_part`: `4.282`
  - `tri_global`: `0.556`
  - `tri_part`: `1.786`

### Epoch 10 首次评估
| Epoch | mAP | R1 | R5 | R10 |
|------|-----|----|----|-----|
| 10 | **37.8%** | **50.2%** | **68.4%** | **73.8%** |

### 对比与判断
1. `Epoch 10` 的首评估数值在合理区间内，没有出现明显异常掉点。
2. `id_part / tri_part` 仍显著高于 global 分支，符合 `PSG+GCN` 早期 part 分支收敛更慢的既有经验。
3. 当前训练曲线平滑，尚未出现需要提前干预的证据。

### 当前判断
- **继续**

### 原因
1. 训练和验证都已正常跑通
2. 首次评估与 warmup 期预期一致
3. 暂无任何需要停训排查的硬异常

---
## [12:09] 检查点 #3

**状态**: ▶️ 运行中

### 训练进度
- 已完成：`Epoch 14/120`
- 当前：`Epoch 15` 已开始
- 最新 ETA：约 `1h39m`

### 中段收敛情况
- `Epoch 11` 末段：
  - `Loss`: `5.804`
  - `Acc`: `0.220`
  - `id_global`: `5.443`
  - `id_part`: `4.039`
  - `tri_global`: `0.522`
  - `tri_part`: `1.604`
- `Epoch 12` 末段：
  - `Loss`: `5.534`
  - `Acc`: `0.224`
  - `id_global`: `5.203`
  - `id_part`: `3.832`
  - `tri_global`: `0.492`
  - `tri_part`: `1.541`
- `Epoch 13` 末段：
  - `Loss`: `5.187`
  - `Acc`: `0.244`
  - `id_global`: `4.930`
  - `id_part`: `3.569`
  - `tri_global`: `0.464`
  - `tri_part`: `1.411`
- `Epoch 14` 末段：
  - `Loss`: `4.959`
  - `Acc`: `0.270`
  - `id_global`: `4.690`
  - `id_part`: `3.378`
  - `tri_global`: `0.444`
  - `tri_part`: `1.405`

### 观察
1. `epoch 11-14` 仍保持稳定下降，没有出现 warmup 后反弹或震荡失控。
2. `id_part` 仍高于 `id_global`，但差距正在按预期缩小，说明 branch 在正常追赶。
3. 当前没有任何迹象表明需要停训或回退。

### 当前判断
- **继续**

### 原因
1. 曲线持续向好
2. 训练速度稳定
3. 准备观察 `epoch 20` 的第二次评估，确认是否延续既有 `exp030a` 早期轨迹
