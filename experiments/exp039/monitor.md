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
| 039a | `cvk_only` | ✅ 完成 | 59.3% | 72.9% | 只用共同可见关键点距离 |
| 039b | `cvk_hybrid` | ✅ 完成 | 61.9% | 73.2% | `global` 与 `common-visible kp distance` 平均融合 |

## 运行前检查
- [x] 默认 config 行为保持不变
- [x] 新 evaluator 已通过本地 dummy smoke test
- [x] `exp037` 仍在运行，遵守用户规则，不主动停训
- [x] 等 GPU 释放后运行 `039a / 039b`

## 039a: `cvk_only`

### [11:11] 结果记录
**状态**: ✅ 完成

| 模式 | mAP | R1 | R5 | R10 |
|------|-----|----|----|-----|
| `cvk_only` | **59.3%** | **72.9%** | **84.1%** | **87.1%** |

**对比**:
- vs `exp035a equal_concat`: mAP `-1.8%` (`59.3 vs 61.1`), R1 `-0.9%` (`72.9 vs 73.8`)
- vs `exp030a-global` 3-seed mean: mAP 近似持平，R1 更高

**观察**:
1. 纯共同可见关键点距离的 **R1 不差**，说明关键点级 pair-specific 支撑确实有信号。
2. 但 mAP 明显落后于 `equal_concat`，说明它还不足以单独承担最终检索距离。
3. 初步判断更符合“关键点距离适合作为补充项，而不是替代 global/equal_concat 主距离”。

**决策**: 继续运行 `039b`
**原因**: 需要验证 `global + cvk` 的 hybrid 是否能保留 `cvk_only` 的 top-1 优势，同时把 mAP 拉回到更接近 baseline 的水平。

## 039b: `cvk_hybrid`

### [11:13] 结果记录
**状态**: ✅ 完成

| 模式 | mAP | R1 | R5 | R10 |
|------|-----|----|----|-----|
| `cvk_hybrid` | **61.9%** | **73.2%** | **85.2%** | **88.5%** |

**对比**:
- vs `exp035a equal_concat`: mAP `+0.8%` (`61.9 vs 61.1`), R1 `-0.6%` (`73.2 vs 73.8`)
- vs `039a cvk_only`: mAP `+2.6%`, R1 `+0.3%`

**观察**:
1. `cvk_hybrid` 明显优于 `cvk_only`，说明 **共同可见关键点距离更适合作为 global 的补充项，而不是替代项**。
2. mAP 相对 `equal_concat` 转正到 `+0.8%`，这是当前对“pair-specific common-support reasoning”最直接的正面证据。
3. R1 没有超过 `equal_concat`，但只差 `-0.6%`，表明 hybrid 带来的收益更偏向整体排序质量（mAP）而不是 top-1 极值。

## exp039 阶段结论
1. **共同可见关键点支撑是真实存在的**：`cvk_only` 的 R1 不差，说明关键点级 pairwise signal 不是噪声。
2. **纯关键点距离不足以替代当前主检索距离**：`cvk_only` mAP 明显低于 baseline。
3. **最合理的使用方式是 hybrid**：`global + cvk` 已在当前单 checkpoint 上给出 `+0.8% mAP` 的正信号。

## 后续动作
- 下一步优先在更干净的 `exp030a` checkpoint 上复核 `cvk_hybrid`，确认这个增益不是 `exp035a` bundled checkpoint 的偶然现象。
