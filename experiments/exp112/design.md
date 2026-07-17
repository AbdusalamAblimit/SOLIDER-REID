# 实验 exp112: High-Confidence Support SCKD（UPDATE_THR=0.7）

## 动机

`exp110` 证明了 `support-complete distillation` 的训练端最小原型可以转正，但增益偏弱。当前最自然的解释之一是 prototype teacher 仍然不够干净：`POSE_SCKD_UPDATE_THR=0.5` 允许中等可见度 keypoint 进入 bank，这会把遮挡噪声、姿态提取误差和局部不稳定特征一并写入 teacher。

本地 `exp111` 正在验证“support 数量门槛”是否重要；为了并行判断 teacher reliability 的另一半来源，本实验专门测试 **support 纯度**。

## 核心假设

如果只允许更高可见度的 keypoint 更新 prototype bank，teacher 会更干净，进而让 `SCKD` 的训练信号更可靠。

更具体地说：

1. `exp110` 的瓶颈之一可能是 bank 写入过宽；
2. 将 `POSE_SCKD_UPDATE_THR` 从 `0.5` 提高到 `0.7` 后，bank coverage 会下降，但 prototype 纯度应提升；
3. 若 teacher purity 比 coverage 更关键，则中后期验证应优于 `exp110`；
4. 若结果变差，则说明当前阶段更缺 coverage，而不是 purity。

## 技术方案

相对 `exp110`，只改一个变量：

- `MODEL.POSE_SCKD_UPDATE_THR: 0.5 -> 0.7`

其余全部保持一致：

- `Swin-Tiny`
- batch size 不变
- `POSE_TEST_FEAT = equal_concat`
- `POSE_SCKD_WEIGHT = 0.5`
- `POSE_SCKD_WARMUP = 20`
- `POSE_SCKD_MIN_COUNT = 1`
- `OUTPUT_DIR = log/occluded_duke/exp112_sckd_up07`

## 对照组

- 主对照：`exp110_sckd`
- 并行参考：`exp111_sckd_min4`

## 预期结果

1. 早期 warmup 曲线应与 `exp110` 一致；
2. `epoch > 20` 后 `sckd` 量级可能略低于 `exp110`；
3. 若 purity 更关键，则 `ep60+` 验证应优于 `exp110`；
4. 若 coverage 更关键，则表现可能回落到基线附近。

## 风险与失败解释

1. `UPDATE_THR=0.7` 可能过严，导致高质量 support 不足；
2. 若结果变差，不意味着 `support-complete` 方向错误，更可能说明：
   - 现阶段 bank 更缺覆盖率；
   - 需要 soft reliability weighting，而不是硬阈值；
   - teacher purity 和 coverage 需要联合建模，而不是单独提纯。
