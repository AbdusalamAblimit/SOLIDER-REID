# exp039: 共同可见关键点检索诊断 — 监控日志

## 实验概述
- **目的**: 验证 `PSG+GCN` branch 的价值是否更适合在检索时用“共同可见关键点距离”来表达
- **基线**:
  - `exp030a-eq` 3-seed mean = `60.73% mAP / 72.57% R1`
  - `exp035a` 单 checkpoint = `61.1% mAP / 73.8% R1`
- **权重来源**: `log/occluded_duke/exp035a_kpw_score/transformer_120.pth`
- **说明**: 这是 retrieval-time diagnostic，不计入训练端创新增益

## 子实验列表
| ID | 模式 | 状态 | mAP | R1 | 备注 |
|----|------|------|-----|----|------|
| 039a | `cvk_only` | 待运行 | — | — | 只用共同可见关键点距离 |
| 039b | `cvk_hybrid` | 待运行 | — | — | `global` 与 `common-visible kp distance` 平均融合 |

## 运行前检查
- [x] 默认 config 行为保持不变
- [x] 新 evaluator 已通过本地 dummy smoke test
- [x] `exp037` 仍在运行，遵守用户规则，不主动停训
- [ ] 等 GPU 释放后运行 `039a / 039b`
