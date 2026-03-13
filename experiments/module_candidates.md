# 模块/方向候选清单（2026-03-13 更新）

## 当前判断标准
- 只保留同时满足“问题定义更清楚”或“机制明显不同于旧模块拼接”的方向。
- 凡是本质上属于 `再学一个权重 / 再加一个小 attention / 再调 pooling` 的候选，默认不进主线。

## 候选 1：共同可见关键点检索（推荐）
**状态**: `推荐进入下一实验`

### 问题定义
- 当前 `PSG+GCN` branch 已证明对 fusion 有价值，但测试时被 `equal_concat` 压成单一向量。
- 对遮挡 ReID 来说，真正需要的是 **query-gallery pair-specific 的共同可见支撑**，而不是固定拼接。

### 机制草案
- 保留 GCN 增强后的 `17 x C` 关键点特征到测试阶段
- 仅在 query/gallery 共同可靠的关键点上计算局部距离
- 再与 global distance 做混合，而不是直接特征拼接

### 为什么值得做
- 问题层面比 `AFF/LKA` 更明确
- 与 BPBreID/KPR/FRT 的主线更一致
- 能直接解释 `exp030a` 为何“global 不涨、fusion 才涨”

## 候选 2：共同可见关键点驱动的 pair-specific fusion
**状态**: `候补`

### 问题定义
- global 和 keypoint branch 的贡献应当依赖于 query-gallery 的共同可见支撑，而不是单图自适应权重。

### 机制草案
- 先算 keypoint overlap / reliability
- 再把它作为 pair-specific 系数去调 global distance 与 local distance 的组合

### 风险
- 如果写法过于接近“质量加权距离”，容易落回已有工作叙事
- 需要先由候选 1 证明 keypoint-level common-support 确实有效

## 候选 2.5：CSGT（Common-Support-Guided Triplet）
**状态**: `推荐进入下一实验`

### 问题定义
- 当前 retrieval-time 证据已经说明：遮挡 ReID 的关键不只是“有没有局部特征”，而是 **batch 内不同 pair 的共同可见支撑并不相同**。
- 但现有 global triplet 仍把所有正负 pair 当成同一可比性假设下的样本来挖 hardest case。

### 机制草案
- 用 skeleton branch 的 `kp_weights` 构造 batch 内 pairwise common-support overlap
- 在 global branch 上增加一条 support-aware triplet：
  - 优先在 overlap 足够高的 pair 上做 hard mining
  - 找不到时回退到标准 mining
- 默认行为不变，完全由 config 开关控制

### 为什么值得做
- 它不是再加一个 branch 模块，而是把 **pair-specific common support** 迁进训练目标
- 相比单纯 test-time `cvk_hybrid`，它更接近训练端创新
- 相比 `exp036` 的逐关键点 triplet，它利用的是 pair 可比性，而不是把每个关键点独立监督一遍

## 候选 3：AFF（Adaptive Feature Fusion）
**状态**: `降级为备选，不作为主线`

### 降级理由
1. 问题定义偏弱，更像在 fixed fusion 上补一个 learnable gate
2. QPM / PAN / RGANet 一类工作早已覆盖质量估计与自适应加权叙事
3. 当前 `exp035b / exp036 / exp037` 已显示 branch 内部权重/损失微调的收益很弱

## 候选 4：继续做 branch 内部 learnable weighting / extra loss
**状态**: `不推荐`

### 不推荐理由
1. `exp035b`: `score*visibility` 负
2. `exp036`: per-kp triplet 负
3. `exp037`: 截至 epoch 100 仍低于 `exp035a` 同期
4. 文献上这类做法也更像局部调参，而不是问题级创新

## 当前结论
- **主线应从“再调 branch 内部模块”切到“如何利用共同可见关键点支撑”。**
- 后续若继续开实验，优先顺序应为：
  1. 共同可见关键点检索诊断
  2. CSGT（训练端 common-support mining）
  3. pair-specific fusion
  4. 若前几者都失败，再回头考虑 AFF 作为纯工程补充
