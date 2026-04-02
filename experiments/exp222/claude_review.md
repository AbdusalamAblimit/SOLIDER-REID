# exp222 GSPB on Small 审查报告

**审查时间**: 2026-04-01
**审查类型**: 快速审查（无新代码，仅 config 参数变更）
**前序审查**: exp220/claude_review.md（GSPB 机制完整审查，已通过）

---

## 审查范围

### a. design.md

**合理性**: 通过。exp220 已在 Tiny 上验证 GSPB (scale=0.05) 改善 maxsim +0.4%。在 Small 上重复验证是标准的 scaling 实验。

**单变量原则**: 通过。相对 exp206r (Small baseline) 只增加 `MODEL.POSE_PART_GRAD_SCALE 0.05`。

**创新门槛**: 不适用 — 这是已验证机制的 backbone 规模验证，不是新创新点。

### b. 代码审查

**无新增/修改代码**。GSPB 相关代码（`model/pose_backbone_model.py` line 115-118, 446-450）与 exp220 审查时完全一致。exp220 审查结论：数学正确（forward 值不变，backward 梯度精确缩放），AMP 安全，无新参数。

### c. 架构兼容性（Tiny → Small）

GSPB 操作是纯元素级的：`feat_map_detached = featmaps[-1].detach() + gs * (featmaps[-1] - featmaps[-1].detach())`。

- 不依赖 channel 维度、spatial 维度或任何 shape
- Swin-Small 的 stage3 输出 channel = 768（与 Tiny 相同，因为 Small 增加的是每 stage 的 block 数而非 embed_dim）
- 即使 channel 不同，element-wise 操作也完全兼容

**通过**。无兼容性问题。

### d. 配置参数

需通过命令行覆盖的参数（相对 exp206r）：

| 参数 | exp206r | exp222 |
|------|---------|--------|
| MODEL.POSE_PART_GRAD_SCALE | 0.0 (default) | 0.05 |

其余参数（TRANSFORMER_TYPE=swin_small, PRETRAIN_PATH, BASE_LR 等）应与 exp206r 保持一致。

**defaults.py**: `POSE_PART_GRAD_SCALE = 0.0`，默认值安全，不影响其他实验。

### e. 优化器

GSPB 不引入任何新的可学习参数。无需检查优化器配置。**通过**。

### f. 对照实验隔离性

| 维度 | exp206r (对照) | exp222 |
|------|---------------|--------|
| POSE_PART_GRAD_SCALE | 0.0 (detach) | 0.05 |
| Backbone | Swin-Small | Swin-Small |
| 其他 | 不变 | 不变 |

**通过**。单变量隔离。

---

## 操作检查项

1. 确认启动命令包含 `MODEL.POSE_PART_GRAD_SCALE 0.05`
2. 确认训练初始化日志输出 `[GSPB] Part branch gradient scale: 0.05`

---

## 汇总

无 Critical / High / Medium / Low 级别问题。
所有代码已在 exp220 审查中逐行验证通过，本次无代码变更。

---

## 结论

**审查通过**。

exp222 是 exp220 (GSPB on Tiny) 的 Swin-Small 规模验证。GSPB 机制是形状无关的元素级操作，Tiny→Small 迁移无任何兼容性风险。唯一变更是命令行参数 `POSE_PART_GRAD_SCALE 0.05`。
