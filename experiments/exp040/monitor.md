# exp040: 基于 exp030a 原始 checkpoint 的 CVK 检索复核 — 监控日志

## 实验概述
- **目的**: 在 `exp030a` 原始 checkpoint 上直接复核 `cvk_hybrid`
- **checkpoint**: `log/occluded_duke/exp030a_psg_gcn/transformer_120.pth`
- **运行方式**:
  - `040a`: `equal_concat`
  - `040b`: `cvk_hybrid`
- **说明**: 这是 retrieval-time verification，不计入训练端创新增益

## 子实验列表
| ID | 模式 | 状态 | mAP | R1 | 备注 |
|----|------|------|-----|----|------|
| 040a | `equal_concat` | ✅ 完成 | 61.1% | 73.7% | 原始 checkpoint 的当前代码口径对照 |
| 040b | `cvk_hybrid` | ✅ 完成 | 61.9% | 73.2% | 在同 checkpoint 上复核 CVK hybrid |

## 运行前检查
- [x] 已确认 `exp030a` checkpoint 存在
- [x] 默认 config 行为保持不变
- [x] 新测试模式已在 `exp039` 中完成 smoke test 与单 checkpoint 诊断
- [x] 运行 `040a`
- [x] 运行 `040b`

## 040a: `equal_concat`

### [11:29] 结果记录
**状态**: ✅ 完成

| 模式 | mAP | R1 | R5 | R10 |
|------|-----|----|----|-----|
| `equal_concat` | **61.1%** | **73.7%** | **85.2%** | **88.0%** |

**观察**:
1. 当前代码下复核出的 `exp030a` checkpoint 结果与历史单 seed 记录一致，说明本轮 evaluator 改动没有破坏原始 `equal_concat` 基线口径。
2. 这给 `040b` 提供了干净直接的同 checkpoint 对照。

**决策**: 继续运行 `040b`
**原因**: 现在可以在同一 checkpoint、同一代码版本下直接判断 `cvk_hybrid` 是否比 `equal_concat` 更优。

## 040b: `cvk_hybrid`

### [11:30] 结果记录
**状态**: ✅ 完成

| 模式 | mAP | R1 | R5 | R10 |
|------|-----|----|----|-----|
| `cvk_hybrid` | **61.9%** | **73.2%** | **85.2%** | **88.6%** |

**对比**:
- vs `040a equal_concat`: mAP `+0.8%` (`61.9 vs 61.1`), R1 `-0.5%` (`73.2 vs 73.7`)
- vs `039b cvk_hybrid` (`exp035a` checkpoint): mAP 相同，R1 相同，R10 `+0.1%`

**观察**:
1. `cvk_hybrid` 在 `exp030a` 原始 checkpoint 上复现了与 `exp039b` 几乎相同的结果，说明此前的正信号不是 bundled checkpoint 偶然现象。
2. 两次实验都稳定表现为 **mAP 转正、R1 小幅回落**，这进一步支持它是一个“整体排序修正项”，而不是 top-1 强驱动项。
3. 当前最稳妥的表述应是：共同可见关键点 reasoning 已经具备 **可复核的单 checkpoint 正向证据**，但还没有到多 seed / 多 checkpoint 的最终结论强度。

## exp040 阶段结论
1. `exp030a` 原始 checkpoint 的当前代码口径 `equal_concat` 基线为 `61.1% mAP / 73.7% R1`。
2. `cvk_hybrid` 在同 checkpoint 上达到 `61.9% mAP / 73.2% R1`，即 `+0.8% mAP / -0.5% R1`。
3. 由于这一结果与 `exp039b` 高度一致，**retrieval-time common-support reasoning 已从“候选想法”升级为“可复核信号”**。

## 后续动作
- 下一步进入 `global : cvk` 权重敏感性验证，确认 `1:1` 是否只是偶然点位，还是较稳的工作区间。
