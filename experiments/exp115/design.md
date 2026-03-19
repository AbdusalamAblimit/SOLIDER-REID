# 实验 exp115: Freeze-Later Reliable SCKD

## 动机

`exp114` 正在验证一个最强版本的假设：bank 在 `warmup` 结束后立刻冻结，看看 teacher non-stationarity 是否就是当前瓶颈。

但如果 `exp114` 失败，原因可能有两种：

1. 持续在线更新确实更好
2. 或者只是 **在 `epoch 20` 冻结得太早**，teacher 还不够成熟

因此需要一个并行单变量来区分这两种解释。

## 核心假设

如果问题主要是 “teacher 在蒸馏阶段一直变硬”，而不是 “冻结本身有害”，那么：

- 让 bank 在 `epoch 30` 冻结
- 即先允许 `10` 个蒸馏 epoch 的共同适配，再固定 teacher

有机会比 `freeze20` 更稳，也更强。

## 技术方案

在 `exp112` 基础上只改一个核心变量：

- `POSE_SCKD_UPDATE_STOP_EPOCH = 30`

保持其他设定不变：

- `POSE_SCKD_UPDATE_THR = 0.7`
- `POSE_SCKD_WARMUP = 20`
- `POSE_SCKD_MIN_COUNT = 1`

## 对照组

1. 主对照: `exp114_sckd_up07_freeze20`
2. 次对照: `exp112_sckd_up07`
3. 全局基线: `exp110_sckd` 与 `exp030a-eq`

## 预期结果

1. 若 `freeze30 > freeze20`：
   - 说明 “teacher stability” 是对的，但 `epoch 20` 冻结太早
2. 若 `freeze20 > freeze30`：
   - 说明一旦进入蒸馏阶段，teacher 就应尽快固定
3. 若二者都不如 `exp112`：
   - 说明问题不在 non-stationarity，而更可能在 teacher weighting / softness 设计

## 风险与失败解释

1. 远程 5060 Ti 较慢，收敛观察会滞后
2. 如果 `freeze30` 与 `exp112` 近乎等价：
   - 说明持续更新并没有带来明显额外伤害
3. 如果 `freeze30` 明显差于 `freeze20`：
   - 说明 “late freeze” 仍太晚，hardening 问题在蒸馏早期就已经发生
