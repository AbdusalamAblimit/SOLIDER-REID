# exp104 PACD 审查报告

## 审查范围
- `experiments/exp104/design.md`
- `experiments/exp104/monitor.md`
- `configs/occluded_duke/pose_psg_gcn_paa_pacd.yml`
- `processor/processor.py`
- `log/occluded_duke/exp104_pacd/train_log.txt`

## 历史修复记录

- 早期实现里曾有 `parallel_aug` 路径 `feat_maps` 未捕获的问题
- 第二轮审查已确认旧版 PACD 存在 heatmap mask、未重归一化、raw MSE 三个关键缺陷
- 当前工作区里的 `processor/processor.py` 已经被继续改成“关键点坐标 + 3x3 邻域 + 归一化池化 + cosine”

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | HIGH | design/log vs 当前代码 | `exp104` 现有日志显然不是当前工作区代码跑出来的：当前代码的 PACD 是 cosine distance，数值应在 `[0, 2]` 左右；但日志里 `pacd` 长期在 `50+`，只能对应早期的 MSE 版本。也就是说，仓库当前代码与 `exp104` 结果已经失配 | 未修复 |
| 2 | CRITICAL | train_log.txt | 已有运行中 `tri_global` 长期卡在约 `0.693`、`pacd` 持续升到 `50+`，说明旧版 PACD 显著破坏了 global metric branch。现有日志不能作为“PACD 有效”的证据 | 未修复 |
| 3 | MEDIUM | design.md vs 当前代码 | design 里写的是“pose-guided spatial mask + MSE distillation”；当前工作区则变成了“关键点邻域 mask + cosine distillation”。即便后续重新训练，它测试的也已经不是原始设计假设 | 未修复 |

## 审查通过项

- `POSE_PACD` 的配置开关和 warmup 入口是清楚的
- 当前工作区版本至少修掉了旧版中最明显的数值病灶

## 结论

❌ **不通过**

`exp104` 现有实验结果无效，且代码与日志已经脱节。后续若要让 Claude 重做，应把当前 PACD 代码单独冻结成新实验编号，再重新跑。
