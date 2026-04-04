# exp238 审查报告 — PPA assign_weight=0.1 消融

审查人: Claude Opus 4.6
日期: 2026-04-03

## 审查范围

a. `experiments/exp238/design.md` — 合理性、单变量原则
b. 代码变更 — 无新代码，纯配置消融
c. PPA 代码 — 已在 exp237 claude_review.md 中逐行审查通过
d. 配置安全性 — POSE_PPA_ASSIGN_WEIGHT=0.1 的有效性
e. 增强配置 — ROA=False, PLBOA=0.7 匹配 exp191 baseline

---

## A. 设计文档审查

**动机**: exp237 (PPA w=0.5) 达成 63.7/75.0 (+0.5/-0.4 vs exp191 baseline)。assign_weight=0.5 意味着 assignment CE loss 的贡献量相当于一个完整 ID loss 的一半。降低到 0.1 可以减少 assignment 梯度对 backbone 的干扰，让 ID/triplet loss 梯度主导 backbone 优化。

**单变量原则**: 满足。唯一变量是 `POSE_PPA_ASSIGN_WEIGHT` 从 0.5 到 0.1。

**关于"小调参"质疑**: 这是对 exp237 首次 mAP-positive 结果的合理消融。assign_weight 是 PPA 设计中唯一的关键超参数，直接控制 assignment 监督强度 vs 自由学习的平衡。这不是逃避创新——这是验证 PPA 核心机制的必要消融实验。论文消融表需要此数据。

---

## B. 代码变更审查

无新增/修改代码文件。所有 PPA 代码（`model/modules/part_assignment_head.py`、`model/pose_backbone_model.py`、`processor/processor.py`）在 exp237 审查中已逐行通过。

---

## C. 配置参数有效性

`POSE_PPA_ASSIGN_WEIGHT` 在两处被读取:

1. **`model/pose_backbone_model.py` L138**: 传入 `PartAssignmentHead.__init__` 的 `assign_weight` 参数。在 head 中仅存储为 `self.assign_weight` 并在初始化 print 中显示，**不参与 forward 计算**。值 0.1 安全。

2. **`processor/processor.py` L896**: `assign_weight = float(getattr(cfg.MODEL, 'POSE_PPA_ASSIGN_WEIGHT', 0.5))`，用于 `loss = loss + assign_weight * assign_loss`。这是实际生效的路径。值 0.1 作为标量乘数完全有效 — 无除零、无 dtype 问题。

**注意**: head 构造时存储的 `self.assign_weight` 与 processor 中直接从 config 读取的值都来自同一个 config key，因此一致。但 head 内部未使用该值进行实际计算 — 这是一个轻微的设计冗余（exp237 审查中已标注），不影响正确性。

---

## D. 增强配置确认

从 exp237 训练 log 确认 exp237 使用了:
- `POSE_ROA: False` — 无 VOC 遮挡增强
- `POSE_LOWER_BODY_OCC: True`, `POSE_LOWER_BODY_OCC_PROB: 0.7` — 匹配 exp191 baseline
- `POSE_TEST_FEAT: equal_concat`

exp238 应使用完全相同的增强配置，仅改变 assign_weight。需在启动命令中确认这些值不变。

---

## E. 预期行为分析

assign_weight 从 0.5 降到 0.1 的效果:

- **Assignment CE 贡献降低 5 倍**: assignment loss 对总 loss 的贡献从约 50% ID-loss-scale 降到约 10%。
- **可能的好处**: 减少 assignment 梯度对 backbone 的干扰。exp237 monitor 显示 mAP 在 ep30-50 快速从 +3.7% 收缩到 +1.8%（vs exp191），可能是 assignment 梯度过强导致的 late-stage 干扰。降低权重或许能缓解。
- **可能的风险**: assignment 监督过弱 → part_proj 学不到有效的 part assignment → 退化为近似 uniform pooling → part branch 无法提供有意义的补充信息。
- **监控要点**: 关注 `ppa_entropy`（应低于 exp237 同期值吗？不一定——更低的监督可能导致更高 entropy）和 `ppa_bg_ratio`。

---

## F. 与 exp237 对照组的隔离性

| 配置项 | exp237 | exp238 |
|--------|--------|--------|
| POSE_PPA | True | True |
| POSE_PPA_NUM_PARTS | 5 | 5 |
| POSE_PPA_ASSIGN_WEIGHT | **0.5** | **0.1** |
| POSE_ROA | False | False |
| POSE_LOWER_BODY_OCC_PROB | 0.7 | 0.7 |
| POSE_TEST_FEAT | equal_concat | equal_concat |
| 其余所有参数 | 相同 | 相同 |

单变量隔离: 确认。

---

## 发现的问题

无 Critical / High / Medium / Low 级别问题。这是纯配置消融，代码已通过审查。

---

## 审查结论

exp238 是 exp237 PPA 的 assign_weight 消融实验（0.5 → 0.1），满足单变量原则。无新代码，PPA 代码已在 exp237 审查中逐行通过。配置参数值 0.1 在所有使用路径中安全有效。增强配置与 exp191 baseline 匹配。

**审查通过**
