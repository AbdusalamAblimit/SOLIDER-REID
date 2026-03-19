# 2026-03-19 文献/代码学习笔记：support-complete 方向

## 背景

`exp107/108` 已经给出两轮较强负证据：
- `pooled hypothesis + counterfactual rerank` 负面
- `per-keypoint / common-support penalty` 仍负面

这说明下一步不能继续停留在 retrieval-time penalty 公式层面，而要回到更本质的问题：
**单张遮挡图本身就是信息不完整的，模型是否能学到“support-complete”的潜在表征？**

---

## 相关工作与真正启发

### 1. BPBreID / KPR：partial observation 下单 embedding 存在理论边界

- BPBreID 仓库在 2025-10 新增了一段非常关键的理论讨论：
  - 全身图 A、上半身图 B、下半身图 C 可能属于同一人
  - 但 `B ↔ C` 没有共同可见证据
  - 因而“所有同 ID 图都应在单 embedding 空间里彼此接近”这个假设在 partial observation 下存在内在悖论
- KPR 则进一步把问题推到 `multi-person ambiguity`，通过 prompt 指定目标人。

**对我们的启发**:
1. 不能再把问题理解成“再做一个更好的全局 embedding”
2. 也不能把问题只理解成“多候选打分公式”
3. 更合理的方向是：让模型学到一个**比当前可见支持更完整、但又不是瞎补全**的 latent target representation

### 2. MVI²P：multi-view integration + propagation 已经证明“同 ID 多视图蒸馏到单图”是合理范式

- MVI²P 的核心是：
  - 训练时利用同 ID 多图整合出 comprehensive representation
  - 再把它传播/蒸馏回单张 occluded image
- 这条线的重要价值不是具体模块，而是它确认了：
  **训练时用 multi-view same-ID 信息，去塑造 test-time single-image 表征，是一条成立的范式。**

**它和我们的 gap**:
1. MVI²P 更偏 holistic / CAM-aware feature map integration
2. 没有显式落在 `per-keypoint / common-support` 粒度
3. 没有处理我们这里非常关键的 pose-noise / duplicate-person artifact

### 3. NFR / FRT / Pose2ID / SGCFR：recover / centralize 往往是对的，但多数停留在 test-time

- NFR、FRT、Pose2ID、以及我们自己的 SGCFR，都说明：
  - 仅靠单图往往不够
  - 利用邻居、生成视角、或 gallery support 做恢复/中心化，经常会带来明显增益
- 但这些方法大多停留在：
  - retrieval-time recovery
  - test-time centralization
  - 或整体 feature map 级恢复

**它们没有真正回答的问题**:
能不能把这种“跨图 support 补全”变成一种**训练时的结构性监督**，让单图编码器本身学会更完整的关键点级 identity support？

---

## 结合当前代码线后的判断

### 我们已有的强证据

1. `PAA` 对多人 query 更有效，说明 scene-level pose 的价值主要在复杂遮挡场景
2. `cvk_hybrid` 和 `SGCFR` 为正，说明 **common-support / recovery** 这类 pair-specific 信号是真实存在的
3. `exp107/108` 为负，说明 **confuser penalty** 不是对的机制
4. `exp091/092/101/105/106` 大多失败，说明：
   - batch 内 same-ID recovery 不够
   - global self-distillation 不够
   - detached pairwise comparator 不够

### 因而最值得赌的新 gap

不是：
- 再做一个新的 test-time rerank
- 再做一个新的 GCN 小模块
- 再做一个新的 visibility / uncertainty weighting

而是：

**Support-Complete Distillation**

即：
- 训练时，从同 ID 的多张图里构造更完整的 `per-keypoint support bank / prototype`
- 再把这个“support-complete teacher”蒸馏回单张 occluded image
- 目标不是直接恢复像素，也不是 retrieval-time 邻居补全
- 而是让单图编码器学到一个更接近“完整身份支持”的潜在关键点表征

---

## 暂定方法草案

### 方法名候选

- `SCKD`: Support-Complete Keypoint Distillation
- `ISPD`: Identity Support Prototype Distillation
- `SKP`: Support-Complete Keypoint Prototypes

### 核心机制

1. 对每个训练 ID、每个 keypoint，维护一个 prototype / memory bank
2. prototype 只由**高可见性样本**更新
3. 当前图若某个 keypoint 低可见：
   - 不强迫它去匹配随机同 batch 图
   - 而是蒸馏到该 ID 的 support-complete prototype
4. 训练目标作用于：
   - per-keypoint feature
   - 以及 recovered/complete pooling 后的 image representation

### 为什么它比已有失败实验更合理

- 比 `LSRM/TTSFR` 更强：不再受 batch 内同 ID 数量限制
- 比 `SGMT/PISD/PACD` 更准：监督落在 `per-keypoint support`，不是 global feature
- 比 `DACHM/DACCM` 更本质：不再依赖 test-time 手工 penalty
- 比 MVI²P 更贴近当前仓库已有优势：
  - pose-aware
  - per-keypoint
  - common-support

---

## 当前最合理的验证顺序

1. 先做 `oracle support bank` 诊断
   - 如果 oracle 上界都很小，这条线立即止损
2. 若 oracle headroom 明显，再做最小训练版：
   - 先只加 prototype distillation
   - 不同时叠加新 decoder / new matcher / test-time trick
3. 若训练版有正信号，再考虑与 `cvk_hybrid` / `SGCFR` 的关系

---

## 暂定结论

在当前所有证据下，**最值得继续推进的新主线**是：

**从 retrieval-time confuser penalty，转向 training-time support-complete distillation。**

它满足：
1. 问题层面有新意：单图不完整，目标是学习 latent support-complete representation
2. 机制层面有新意：same-ID multi-view support bank → single-image keypoint distillation
3. 证据层面可讲清楚：可先做 oracle，再做最小训练版，再和 SGCFR/MVI²P 对照
