# 实验 exp107: DACHM（Duplicate-Aware Counterfactual Hypothesis Matching）

## 动机
- `exp030a` 是当前主基线；`exp066` 证明 scene-level pose 注入对多人图更有效，但仍缺少更强的问题定义。
- 本地诊断显示：
  - `target_person_idx` 重排实际只改变极少数样本，说明“把 target 选对”不是主矛盾。
  - `cvk_hybrid` 的 mAP 增益在多人 query 上更明显，说明 pair-specific reasoning 确实有价值。
  - 高人数样本中混有明显的重复检测伪多人，不能把所有 `num_persons>=2` 都视作真实多人歧义。
- 因此下一步需要验证的新问题不是“再做一个 target-aware adapter”，而是：
  **在真实多人重叠 + 重复检测噪声并存时，query-gallery pair 是否需要 duplicate-aware 的反事实候选匹配。**

## 核心假设
1. 当前 `person0 -> person0` 的 target-target 比较不足以覆盖所有困难样本。
2. 误检索里存在一类系统性错误：
   - query 的目标人更像 gallery 的非目标候选；
   - 或 gallery 的目标人更像 query 的非目标候选。
3. 如果先在单图内去除明显重复检测，再对 query-gallery top-K 候选施加反事实 confuser margin 约束，应该能提升排序质量，且收益应主要来自真实多人重叠子集，而不是伪多人噪声。

## 技术方案

### 1. 多候选特征提取
- 基于现有 checkpoint 和 `skeleton_head`
- 对每张图的每个 detected person：
  - 通过 person reordering 把该候选移到 index 0
  - 重新走一次 `skeleton_head`
  - 得到该候选的 pooled skeleton feature

### 2. Duplicate-aware 去重
- 不直接相信 pose extractor 输出的 `num_persons`
- 基于关键点导出的候选框和关键点几何相似度，判定两个候选是否为“同一人的重复检测”
- 去重后：
  - target 候选固定保留
  - 重复的 distractor 不进入 confuser 集合

### 3. Counterfactual Hypothesis Matching
- 基线距离：`equal_concat` 的欧氏距离
- 对每个 query 的 top-K gallery 候选，额外计算：
  - `d_tt`: query target vs gallery target
  - `d_q_gd`: query target vs gallery distractors 的最小距离
  - `d_qd_g`: query distractors vs gallery target 的最小距离
  - `support_gap = min(d_q_gd, d_qd_g) - d_tt`
- 反事实重排：
  - 若 `support_gap` 小，说明该 pair 的 target-target 优势不足，属于高歧义 pair
  - 用 `support_gap` 对 top-K 基线距离做 margin-based 调整

### 4. 评测设置
- 主基线：`exp030a` `equal_concat`
- 支持性复核：`exp066` `equal_concat`
- 对照：
  - `base equal_concat`
  - `raw counterfactual`（不去重）
  - `DACHM`（先去重，再做 counterfactual）
- 子集：
  - `single`
  - `multi`
  - `n=2`
  - `n=3`
  - `n>=4`
  - `duplicate-suspect multi`
  - `clean multi`

## 对照组
- 主对照：`exp030a-eq = 61.1% mAP / 72.7% R1`（seed1234）
- 机制近邻对照：`cvk_hybrid`
- 本实验不改训练，只验证 retrieval-time 机制是否有独立 headroom

## 预期结果
- 如果成立：
  - `DACHM` 相对 `equal_concat` 至少有稳定正 mAP 信号
  - 去重版优于不去重版，特别是在 `duplicate-suspect multi` 子集
  - `clean multi` 子集收益高于 `single`
- 如果失败：
  - 说明“反事实 confuser”虽存在，但判别力不够稳定
  - 或重复检测噪声太重，training-free 机制难以从 noisy hypotheses 中提纯有效信号

## 风险与失败解释
1. confuser margin 在正确 pair 和错误 pair 上分布重叠太大，无法稳定改善排序
2. 关键点导出的 duplicate heuristic 不够准确，把真实高重叠人误合并
3. 多候选 skeleton feature 本身仍不够区分 target / distractor
4. 该方向若只有 test-time 正信号、但没有训练端落点，则只能作为诊断证据，不能直接当论文主贡献
