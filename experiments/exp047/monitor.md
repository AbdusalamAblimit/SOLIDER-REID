# exp047: Common-Support-Guided Triplet (CSGT) — 监控日志

## 实验目标
- **目的**: 把 retrieval-time 的 common-support 信号迁到训练端 triplet mining
- **配置**: `configs/occluded_duke/exp047_csgt_triplet.yml`
- **输出目录**: `log/occluded_duke/exp047_csgt_triplet`

## 启动前检查
- [x] Backbone 固定为 `Swin-Tiny`
- [x] batch size 保持 `64`
- [x] 相对 `exp030a` 只新增 `POSE_CSGT` 相关开关
- [x] 默认代码路径保持不变，开关关闭时完全回退 baseline
- [x] `POSE_CSGT` 不依赖 `POSE_KP_TRIPLET` 才能生效
- [x] `CSGT` 作为独立损失项额外叠加，不并入已有 global/part 混合权重
- [x] 当前先完成代码接入与设计，不抢占正在运行的 `exp046`
- [x] 最终评测口径已固定：
  - `equal_concat` 主汇报
  - `global` 机制对照
  - 训练中的 `concat_scaled` 只作为监控口径

---
## [准备完成] 当前状态

### 计划命令
- `/root/miniconda3/envs/solider-reid/bin/python -u train.py --config_file configs/occluded_duke/exp047_csgt_triplet.yml`

### 训练后必须补的评测
- `exp047a-eq`:
  - `/root/miniconda3/envs/solider-reid/bin/python -u test.py --config_file configs/occluded_duke/exp047_csgt_triplet.yml MODEL.POSE_TEST_FEAT equal_concat TEST.WEIGHT ./log/occluded_duke/exp047_csgt_triplet/transformer_120.pth OUTPUT_DIR ./log/occluded_duke/exp047a_csgt_eq`
- `exp047b-global`:
  - `/root/miniconda3/envs/solider-reid/bin/python -u test.py --config_file configs/occluded_duke/exp047_csgt_triplet.yml MODEL.POSE_TEST_FEAT global TEST.WEIGHT ./log/occluded_duke/exp047_csgt_triplet/transformer_120.pth OUTPUT_DIR ./log/occluded_duke/exp047b_csgt_global`

### 当前判断
- **等待 GPU 空档后启动**

### 原因
1. `exp046` 正在重建 `seed2024` checkpoint，按当前优先级应继续保持。
2. `exp047` 的代码和配置可先准备好，等 `exp046` 进入更稳定阶段或结束后立即开跑。

---
## [校正] 关键接线修复

### 修复内容
1. `POSE_CSGT` 现在可独立拿到 `kp_data`，不再依赖 `POSE_KP_TRIPLET=True`
2. `CSGT` 现在作为独立损失项额外叠加，不再被 `wt_g` 隐式打折
3. `POSE_KP_TRIPLET=False` 时，不会误走逐关键点 triplet 分支

### 最小验证
- `py_compile` 已通过：
  - `processor/processor.py`
  - `loss/make_loss.py`
  - `config/defaults.py`
- 随机张量 smoke test 已通过：
  - `POSE_CSGT=True`
  - `POSE_KP_TRIPLET=False`
  - `tri_csgt` 能正常出现在 `loss_details`
  - `tri_kp` 不会误触发

### 当前判断
- **可启动**

### 原因
1. 之前的“空跑 baseline”风险已排除。
2. 代码与 `design.md` 的损失定义现已一致。

---
## [19:03] 检查点 #1

**状态**: ▶️ 已启动

### 训练进度
- 已进入：`Epoch 1`
- 最新位置：`Iter 40/227`

### 当前局部训练状态
- `Epoch 1 Iter 20/227`：
  - `Loss`: `35.772`
  - `Acc`: `0.000`
  - `id_global`: `6.555`
  - `id_part`: `6.683`
  - `tri_global`: `14.249`
  - `tri_part`: `16.638`
  - `tri_csgt`: `13.710`
- `Epoch 1 Iter 40/227`：
  - `Loss`: `30.971`
  - `Acc`: `0.000`
  - `id_global`: `6.556`
  - `id_part`: `6.681`
  - `tri_global`: `11.848`
  - `tri_part`: `14.060`
  - `tri_csgt`: `11.399`
  - `csgt_pos_overlap`: `0.666`
  - `csgt_neg_overlap`: `0.708`
  - `csgt_pos_fallback`: `0.800`
  - `csgt_neg_fallback`: `0.100`

### 观察
1. 训练已正常启动，数据、模型、优化器和 pose 路径均完成加载。
2. `tri_csgt` 已在首轮日志中出现，说明这次不是“配置开了但实际没生效”的空跑 baseline。
3. `csgt_pos/neg_overlap` 与 fallback 统计也已打印，后续可以直接据此判断 common-support mining 是否真的在工作。

### 当前判断
- **继续**

### 原因
1. 首轮损失下降正常，暂未见 `NaN / OOM / 爆 loss`。
2. 关键新增训练项已经实际接线生效，具备继续观察价值。
