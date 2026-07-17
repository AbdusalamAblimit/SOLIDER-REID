# 实验 exp044: 重建 `exp030a seed42` checkpoint

## 动机
- 当前 `exp039-043` 已经把 `cvk_hybrid` 的单 checkpoint 证据链补得比较完整。
- 但要继续往多 seed 复核推进，需要 `exp030a` 不同 seed 的 checkpoint。
- 本地现状：
  - `4090_log/multiseed/exp030a_psg_gcn_seed42/` 与 `seed2024/` 仅保留日志
  - 对应 `transformer_120.pth` 不在工作区
- 因此下一步不是继续调参，而是先补回可复用资产。

## 核心假设
- 若按原始 `exp030a` 设定重建 `seed42`，应得到与历史多 seed 文档同量级的结果。
- 即使最终数值存在微小波动，只要 checkpoint 可复用，后续就能做 `cvk_hybrid` 的多 seed 复核。

## 技术方案
- 固定训练配置为 `exp030a` 主基线设定
- 唯一变量：
  - `SOLVER.SEED = 42`
  - 独立 `OUTPUT_DIR`
- 训练完成后再补：
  - `equal_concat`
  - `cvk_hybrid`

## 对照组
- 参考历史记录：`exp030a` 的 seed42 多 seed 结果已在既有文档中出现
- 当前实验的目标不是刷新指标，而是恢复 checkpoint 资产；最终对照数字以本次训练日志和后续测试日志为准

## 预期结果
- 最理想：顺利训练并产出 `transformer_120.pth`
- 次优：即使最终指标与历史略有波动，也能为后续 `cvk_hybrid` 多 seed 复核提供可复用权重

## 风险与失败解释
1. 这是资产恢复实验，不应被当作新方法结果。
2. 若与历史日志有偏差，需先确认环境/代码是否与当时一致。
3. 若训练时间过长，也应保持后台运行并持续记录。
