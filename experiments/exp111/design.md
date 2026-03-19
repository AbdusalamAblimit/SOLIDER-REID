# 实验 exp111: Reliable-Support SCKD（MIN_COUNT=4）

## 动机

`exp110` 证明了训练端的 `support-complete prototype distillation` 能在单 seed 上转正，但增益仍然偏弱。当前实现里 `POSE_SCKD_MIN_COUNT=1`，意味着某个 identity 的某个 keypoint 只要出现过一次 high-visibility 观测，就会被当作 teacher 使用。这种 teacher 过于宽松，容易把偶然噪声、姿态抽取误差、遮挡下的不稳定局部都写入 bank。

用户也明确强调了当前阶段应优先探索“真正有效且足够支撑论文的创新点”，而不是提前做多 seed 收尾。因此下一步最合理的单变量推进，不是重复验证，而是把 `support-complete` 主线里的关键机制继续做实。

## 核心假设

如果要求 prototype 至少由多个支持样本共同支撑，teacher 的可靠性会提高，进而让 `SCKD` 从“弱正向”变成“更清楚的正向”。

更具体地说：

1. `exp110` 的瓶颈更像是 teacher 质量，而不是 distillation 思路本身；
2. `MIN_COUNT=1` 太容易让 bank 里混入 noisy teacher；
3. 将 `POSE_SCKD_MIN_COUNT` 提高到 `4` 后，蒸馏覆盖率会下降，但 teacher 可信度应提高；
4. 若假设成立，最终验证应优于 `exp110`。

## 技术方案

相对 `exp110`，只改一个变量：

- `MODEL.POSE_SCKD_MIN_COUNT: 1 -> 4`

其余保持一致：

- `Swin-Tiny`
- batch size 不变
- `POSE_TEST_FEAT = equal_concat`
- `POSE_SCKD_WEIGHT = 0.5`
- `POSE_SCKD_WARMUP = 20`
- `OUTPUT_DIR` 独立为 `log/occluded_duke/exp111_sckd_min4`

## 对照组

- 主对照：`exp110_sckd`
- 更上游基线：`exp030a-eq`

## 预期结果

1. 训练稳定性应与 `exp110` 基本一致；
2. `sckd` 量级可能略降，或触发更晚；
3. 若 teacher reliability 是当前瓶颈，则 `ep60+` 验证应逐步高于 `exp110`；
4. 若覆盖率损失过大，则可能表现为中后期增益变弱或回到基线。

## 风险与失败解释

1. `MIN_COUNT=4` 可能过严，导致大量 keypoint 在训练前半段无法触发蒸馏；
2. 若结果变差，不代表 `support-complete` 主线错误，更可能说明：
   - 该阈值过高；
   - 需要更平滑的 reliability 权重，而不是硬阈值；
   - 问题在 coverage，而不是 teacher purity。
