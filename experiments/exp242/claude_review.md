# exp242 Claude Review: PPA + GCN 双分支 on Small

## 审查范围

a. design.md 合理性与单变量原则
b. 配置安全性（Tiny→Small 变更点）
c. OOM 风险评估
d. 与 exp241 (Tiny) 代码一致性（无新代码）

---

## a. design.md 合理性

动机清晰：exp241 (PPA+GCN on Tiny) 取得 +0.5/-0.1 的最佳综合结果，需在 Small 上验证跨 backbone 泛化性。对照组设置合理：

- exp241 (Tiny PPA+GCN): 63.7/75.3 — 同方法不同 backbone
- exp206r (Small GCN-only): 70.6/82.6 — 同 backbone 不同方法

**单变量原则**: 本实验相对 exp240 (Small PPA) 是单变量——添加 GCN。相对 exp241 (Tiny PPA+GCN) 也是单变量——换 backbone。两个视角都满足单变量隔离。

**创新门槛**: 这是 backbone scaling 验证实验，不是新创新。作为论文中的 cross-backbone 消融证据，是合理的实验类型。

## b. 配置安全性

exp242 需要相对 exp241 做以下变更（全部通过命令行覆盖即可）：

| 参数 | exp241 (Tiny) | exp242 (Small) | 状态 |
|------|------|------|------|
| TRANSFORMER_TYPE | swin_tiny_patch4_window7_224 | swin_small_patch4_window7_224 | 必须改 |
| PRETRAIN_PATH | pretrained/swin_tiny.pth | pretrained/swin_small.pth | 必须改 |
| BASE_LR | 0.0008 | 0.0004 | 必须改（Small 标准） |
| TEST.IMS_PER_BATCH | 256 | 128 | OOM 缓解 |
| OUTPUT_DIR | exp241_ppa_gcn_tiny | exp242_ppa_gcn_small | 必须改 |

其余配置保持不变：
- POSE_PPA: True, POSE_PPA_ASSIGN_WEIGHT: 0.5 ✓
- POSE_SKELETON_GCN: True, POSE_GCN_LAYERS: 2, POSE_GCN_HIDDEN: 256 ✓
- POSE_ADDITIVE_ADAPTER: True (PAA) ✓
- POSE_OA_SD: True ✓
- POSE_ROA: False ✓
- POSE_LOWER_BODY_OCC: True, PLBOA_PROB: 0.7 ✓
- POSE_BACKBONE_PSG: True ✓
- IMS_PER_BATCH (train): 64 ✓

**BASE_LR 注意**: exp241 Tiny 用 0.0008，Small 实验（exp206r, exp230, exp240）统一用 0.0004。这是惯例做法，exp242 必须使用 0.0004。

**特征维度**: Swin-Small embed_dims=96, Stage 3 输出 768 维——与 Tiny 相同。PPA (PartAssignmentHead)、GCN (SkeletonGCN)、PSG 模块的 `feat_dim` / `feat_channels` 参数都基于 `self.in_planes`（768），无需修改。

**Stage 3 blocks 数量**: Small 有 18 blocks（Tiny 有 6）。PSG 和 PAA 模块会自动按 block 数创建（循环 `range(len(stage.blocks))`），无需手动配置。模块数增加 3 倍是自动的。

## c. OOM 风险评估

Small (18 blocks Stage 3) + PPA (non-detached) + GCN (detached) + OA-SD (EMA copy) + PAA (18 modules):

- **训练显存**: exp240 (Small PPA, 无 GCN) 在 3090 24GB 上未 OOM。exp242 额外添加 GCN 分支（detached 采样 + 2-layer GCN + 1 个分类器 + 1 个 BN head），增量约 ~2-3M 参数 + 少量 activation memory（因 detached 不保留梯度图）。GCN 的 detach 特性意味着它不增加 backward 显存。**风险低**。
- **评估显存**: TEST.IMS_PER_BATCH 从 256 降至 128，与 exp240 一致。足够安全。
- **OA-SD EMA 模型**: 只存参数不存梯度，增量与模型参数量线性。Small vs Tiny 多约 2x 参数（50M vs 28M），EMA 副本多占 ~88MB。这在 24GB 中可容纳。

**结论**: OOM 风险可控。exp240 已验证 Small PPA 可在 3090 上训练，GCN detached 分支增量很小。

## d. 无新代码确认

设计文档明确声明"与 exp241 相同，换 Small backbone"。exp241 的代码已在 claude_review.md 中通过审查（梯度隔离、输出结构、loss 处理、测试路径、OA-SD 兼容性全部验证通过）。

本实验不引入任何新代码或新模块，仅通过命令行参数切换 backbone。已审查的代码路径对 backbone 尺寸无硬编码依赖（维度通过 `self.in_planes` 动态获取，模块数通过循环自动适配）。

## 潜在注意点（非 blocking）

1. **PSG+PAA 模块数**: Small Stage 3 = 18 blocks → 18 个 PSG + 18 个 PAA 模块（Tiny 只有 6+6）。参数增量约 3x，但每个模块很小（PSG ~17K, PAA ~50K），总增量约 ~1.2M。不影响正确性。
2. **训练时间**: Small 约为 Tiny 的 2-2.5 倍，预计 6-8 小时。
3. **预期收益**: exp240 (Small PPA-only) 结果为 -0.1/-0.8（基本中性），GCN 在 Tiny 上改善了 R1 (+0.3)。Small 上 PPA+GCN 可能比 PPA-only 好，但整体相对 exp206r 可能仍是小幅变化。

## 结论

配置变更清晰且完整，单变量隔离良好，OOM 风险可控（TEST.IMS_PER_BATCH=128 已做预防），无新代码引入。

**审查通过**
