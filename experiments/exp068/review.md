# exp068 RR-PAA 审查报告

## 审查范围
- `experiments/exp068/design.md`
- `configs/occluded_duke/pose_psg_gcn_rrpaa.yml`
- `model/modules/pose_additive_adapter.py`
- `log/occluded_duke/exp068_rrpaa/train_log.txt`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | HIGH | pose_additive_adapter.py | RR-PAA 的“routing mask”并没有形成设计里所说的可见/遮挡分离。基于真实 heatmap 抽样统计，Stage 3 上 `body_conf` 平均仅约 `0.52`，`occlusion_mask = 1 - body_conf` 基本落在 `0.34~0.50`，更像对 adapter 做近乎均匀的缩放，而不是“只在低置信度区域激活” | 未修复 |
| 2 | MEDIUM | design.md | 设计写的是 suppress-and-complete 式 routed injection，但当前实现只能支持“软衰减版 PAA” | 未修复 |

## 审查通过项

- `POSE_PAA_ROUTED=True` 已正确传到 `PoseAdditiveAdapter(routed=True)`
- 不会污染 baseline，默认仍关闭
- 日志显示训练路径稳定

## 结论

❌ **不通过**

`exp068` 当前没有真正实现“Reliability-Routed”。它测试到的是一个被全局软缩放后的 PAA，而不是面向遮挡区域的 selective completion。
