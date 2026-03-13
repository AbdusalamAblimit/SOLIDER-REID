# Paper 15: 共同可见支撑训练端化的文献缺口总结
**来源**:
- KPR (ECCV 2024): https://arxiv.org/abs/2407.18112
- BPBreID (WACV 2023): https://arxiv.org/abs/2211.03679
- FRT (TIP): https://github.com/xbq1994/Feature-Recovery-Transformer
- QPM: https://arxiv.org/search/?query=Quality-aware+Part+Models+for+Occluded+Person+Re-identification&searchtype=all&source=header
- RPTM (adjacent metric learning): https://arxiv.org/search/?query=RPTM%3A+Reidentification+by+Relation+Preserving+Triplet+Mining&searchtype=all&source=header
**阅读日期**: 2026-03-13

## 这轮学习真正回答了什么

### 1. 巨人们到底有没有做过“共同可见支撑”？
- **做过。**
- KPR / BPBreID / QPM 都已经非常明确地把问题定义成：
  - 不应该无条件比较整个人
  - 而应该只比较 query-gallery 的共同可见局部
- FRT 进一步把这个问题推进成 retrieval-time feature recovery。

### 2. 他们主要把这件事放在哪里？
- **主流落点仍然是匹配端 / 检索端。**
- 也就是：
  - part-visible masked distance
  - pair-specific quality weighting
  - gallery-assisted retrieval-time reasoning
- 不是完全没人碰训练，但训练侧多数仍在：
  - part visibility supervision
  - part triplet
  - learnable weighting
  - part-quality estimation

### 3. 他们没解决什么？
- 在我们当前最相关的代码条件下，还缺一个更精确的机制：
  **如何把 keypoint/skeleton branch 提供的 pair-specific common-support 信号直接写进训练目标，而不是只在测试时拿来算距离。**

更具体地说，现有主流工作里缺少下面这个组合：
1. backbone 仍以标准 ReID 方式训练
2. branch 保留 keypoint-level 结构证据
3. 训练时用这份结构证据去约束 batch 内的 pair / triplet 选择
4. 不依赖 parsing mask，也不依赖 prompt 输入

### 4. 我们还能争什么？
- 不能争：
  - `visibility weighting`
  - `quality-aware fusion`
  - `part-only masked distance`
  - “共同可见区域很重要” 这种泛化结论
- 可以争：
  - **partial observation 下，triplet 把所有正负 pair 一视同仁是有问题的**
  - **共同可见支撑应进入训练期 pair mining，而不是只在检索期做后验修正**
  - **keypoint/skeleton branch 可以作为 pairwise common-support 的训练期特权信号**

## 当前最值得试的训练端候选

### CSGT: Common-Support-Guided Triplet
- 核心问题：
  标准 triplet 默认把 batch 内所有正负 pair 放在同一可比性假设下，但遮挡 ReID 中不同 pair 的共同可见支撑明显不同。
- 核心机制：
  1. 用 `kp_weights` 构造 batch 内 pairwise common-support overlap
  2. 只在 overlap 足够高的 pair 上优先做 hard mining
  3. 把这条 support-aware triplet 作为额外训练约束加到 global branch

### 为什么它比“再学一个权重”更像新方向
1. 它重新定义的是 **哪些 pair 应该被强约束**，不是再给 feature 融合乘一个系数。
2. 它把 `cvk_hybrid` 的 test-time 观察提升成了 train-time 机制。
3. 它和已有 parsing/part 方法不同，依赖的是我们已验证有效的 skeleton branch，而不是重新造一个 part framework。

## 这件事够不够支撑 B 类论文主贡献？
- **目前是候选，不是结论。**
- 它有潜力，因为它同时满足：
  1. 问题层面：partial observation 下 pair 可比性不一致
  2. 机制层面：common-support-guided mining
  3. 证据层面：可直接设计 `exp030a` 对照、对照 `exp036`、再看是否削弱 test-time `cvk_hybrid` 需求
- 但要真正站住，还必须满足至少两件事：
  1. 训练结果本身要稳定优于 `exp030a`
  2. 它要能解释为什么 retrieval-time CVK 还需要或不再需要

## 当前结论
- 这轮学习支持的不是“继续深挖 test-time 权重”。
- 更高价值的下一步是：
  **把 common-support 从检索端诊断，推进成训练端 pair-mining 机制。**
