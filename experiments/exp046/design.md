# 实验 exp046: 重建 `exp030a seed2024` checkpoint

## 动机
- `exp045` 已在重建的 `seed42` checkpoint 上复核到：
  - `equal_concat = 60.2% / 72.7%`
  - `cvk_hybrid = 61.1% / 73.2%`
  - 差距 = `+0.9% mAP / +0.5% R1`
- 这说明 `cvk_hybrid` 的正 mAP 信号已经至少跨两个 checkpoint 成立。
- 但若要进一步靠近真正的多 seed 证据，还缺 `seed2024` 的可复用 checkpoint 资产。
- 本地现状仍是只有日志，没有对应 `transformer_120.pth`，因此应继续先补资产，而不是开新调参。

## 核心假设
- 若按 `exp030a` 原始训练配方重建 `seed2024`，应能得到与历史多 seed 文档同量级的结果。
- 即使最终数值与旧记录有轻微偏差，只要 checkpoint 可复用，就可以继续完成 `equal_concat / cvk_hybrid` 的第三个 checkpoint 复核。

## 技术方案
- 固定训练配置为 `exp030a` 主基线设定
- 唯一变量：
  - `SOLVER.SEED = 2024`
  - 独立 `OUTPUT_DIR`
- 训练完成后再补：
  - `equal_concat`
  - `cvk_hybrid`

## 对照组
- 参考历史记录：`exp030a` 的 seed2024 多 seed 结果已在既有文档中出现
- 当前实验的目标仍是恢复 checkpoint 资产；最终对照数字以本次训练日志和后续测试日志为准

## 预期结果
- 最理想：顺利训练并产出 `transformer_120.pth`
- 次优：即使最终指标与历史略有波动，也能为第三个 checkpoint 的 CVK 复核提供可复用权重

## 风险与失败解释
1. 这是资产恢复实验，不应被当作新方法结果。
2. 若与历史日志有偏差，需先确认环境/代码是否与当时一致。
3. 训练期间仍需按短轮询节奏记录，不能放着不看。
