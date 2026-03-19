# 实验 exp108: DACCM（Duplicate-Aware Counterfactual Common-Support Matching）

## 动机
- `exp107 DACHM` 已证伪：在 **pooled person embedding** 上做 duplicate-aware counterfactual rerank 整体负面。
- 但这并不等于 `target/distractor ambiguity` 这条问题线被否定，因为已有正信号都发生在更细粒度：
  - `cvk_hybrid`
  - `SGCFR`
- 因此，`exp108` 的核心不是继续调 `exp107` 的公式，而是把同一问题重新落在 **per-keypoint / common-support** 粒度：
  **只有在关键点可见性和 common-support 层面，target-target 与 target-distractor 的差异才可能被稳定表达。**

## 核心假设
1. `target/distractor ambiguity` 如果存在可利用信号，它应该首先体现在 per-keypoint common-visible distance 上，而不是 pooled person embedding 上。
2. 对 query-gallery top-K 候选，若存在：
   - `query target ↔ gallery distractor` 或
   - `query distractor ↔ gallery target`
   的 common-support 距离比 `target ↔ target` 更小，则该 pair 应被惩罚。
3. 与 `exp107` 不同，duplicate-aware pruning 在 per-keypoint 层面才可能真正发挥作用，因为 duplicate detection 与 visibility/common-support 是同一层面的结构信息。

## 技术方案

### 1. 多候选关键点特征提取
- 基于 `exp030a` checkpoint
- 对每张图的每个 detected person：
  - reorder 到 index 0
  - 重新走 `skeleton_head(return_cls=False)`
  - 取 `kp_feats` 与 `kp_weights`

### 2. Duplicate-aware pruning
- 用关键点导出的 bbox IoU + 关键点几何相似度识别重复检测
- raw hypotheses 与 deduped hypotheses 同时保留，作为对照

### 3. 基线距离
- 主基线不是 `equal_concat`，而是 **`cvk_hybrid` on exp030a**
- 即：`global distance + target-target common-visible kp distance`

### 4. Counterfactual common-support penalty
- 对每个 query 的 top-K gallery 候选，计算：
  - `d_tt`: query target vs gallery target 的 common-visible kp distance
  - `d_q_gd`: query target vs gallery distractors 的最小 common-visible kp distance
  - `d_qd_g`: query distractors vs gallery target 的最小 common-visible kp distance
  - `support_gap = min(d_q_gd, d_qd_g) - d_tt`
- 只使用 **penalty-only** 版本：
  - 当 `support_gap < 0` 时，说明 confuser 比 target-target 更占优，增加距离惩罚
  - 不再像 `exp107` 那样对“安全 pair”做奖励

## 对照组
- 主对照：`exp030a cvk_hybrid`
- 机制对照：
  - `base_cvk_hybrid`
  - `raw_daccm_penalty`
  - `daccm_penalty`

## 预期结果
- 如果成立：
  - `daccm_penalty` 至少在 mAP 上优于 `base_cvk_hybrid`
  - 收益主要来自 `multi / clean multi` 子集
  - dedup 版优于 raw 版
- 如果失败：
  - 说明 ambiguity 这条 retrieval-time 线即使下沉到 per-keypoint/common-support，也还不足以形成稳定可用的排名信号
  - 后续若继续该方向，需要进入训练端，而不是继续 test-time 公式打补丁

## 风险与失败解释
1. `cvk_hybrid` 已经吃掉了 target-target 的主要 common-support 信号，confuser penalty 额外增益不足
2. duplicate-aware heuristic 仍然不够准，混淆真实重叠人与重复检测
3. 即使粒度足够细，当前 `kp_feats` 表示也没有把 distractor-confuser 和 target-target 稳定分开
