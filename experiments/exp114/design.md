# 实验 exp114: Freeze-After-Warmup Reliable SCKD

## 动机

`exp112/113` 表明当前 `support-complete` 主线的主要瓶颈已不再是 `count gating`，而更像是 **online teacher 持续变硬**：

1. `UPDATE_THR=0.7` 相比 `exp110` 只有弱正向，说明单纯提升写入纯度还不够。
2. `exp113` 诊断显示：
   - `sckd_pairs / active_ratio / elig_ratio / proto_conf` 基本稳定
   - `proto_count` 持续增长
   - `sckd_cos` 持续下降
   - raw `sckd` 随之上升
3. 这说明当前更可能的问题是：
   **teacher bank 在 student 开始蒸馏后仍持续强化，造成 non-stationary target / teacher hardening。**

## 核心假设

如果在 `warmup` 结束时冻结 prototype bank，只保留一个稳定 teacher，再让 student 从这个固定 teacher 蒸馏，那么：

1. `sckd_cos` 不会继续明显恶化
2. raw `sckd` 会比 `exp113` 更稳定
3. 验证结果有机会优于 `exp112` 的弱正向区间

## 技术方案

在 `exp112` 基础上只改一个核心变量：

- 新增 `POSE_SCKD_UPDATE_STOP_EPOCH`
- 当 `epoch > POSE_SCKD_UPDATE_STOP_EPOCH` 时：
  - `sckd_bank.compute_loss()` 仍正常使用
  - `sckd_bank.update()` 停止

本实验设置：

- `POSE_SCKD_UPDATE_THR = 0.7`
- `POSE_SCKD_UPDATE_STOP_EPOCH = 20`

解释：

- `epoch 1-20` 仍允许 bank 累积 support
- `epoch 21+` 正式开启蒸馏时，teacher 固定
- 这正好对应 “freeze-after-warmup”

## 对照组

1. 主对照: `exp112_sckd_up07`
   - 只提升写入纯度，不冻结 teacher
2. 次对照: `exp110_sckd`
   - 默认 `update_thr=0.5`，bank 全程在线更新
3. 全局基线: `exp030a-eq seed1234`

## 预期结果

如果假设成立，应看到：

1. `ep30/40` 起验证不再只是弱波动，而是比 `exp112` 更稳定
2. `sckd_cos` 在 `epoch 20+` 后不再持续下降
3. 若最终仍只和 `exp112` 打平，则说明：
   - 问题不在 “teacher 是否变化”
   - 而更可能在 teacher 的 soft weighting / target hardness 设计

## 风险与失败解释

1. 若结果变差：
   - 说明固定 teacher 太早，bank 在 `epoch 20` 时仍不够成熟
2. 若结果完全等价：
   - 说明 non-stationary teacher 不是主要瓶颈
3. 若 `sckd_cos` 稳住但验证不涨：
   - 说明当前 distillation target 本身不够有信息量，只是更稳定
