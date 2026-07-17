# Paper 1: KPR - Keypoint Promptable Re-Identification
**来源**: ECCV 2024  
**论文**: https://arxiv.org/abs/2407.18112  
**代码**: https://github.com/VlSomers/keypoint_promptable_reidentification  
**阅读日期**: 2026-03-13

## 这篇工作真正解决什么
- 不是简单“再加一个 pose 分支”，而是明确提出 **Multi-Person Ambiguity (MPA)**：
  单个行人框里出现多人时，模型首先不知道“要重识别的是谁”。
- 因此 KPR 的问题定义是：**如何让 ReID 在多人遮挡时仍然有显式目标指向能力**。
- 论文把 prompt 设计成可选项：无遮挡/无歧义时可以不用 prompt，有歧义时再加关键点 prompt。

## 代码里真正怎么做

### 1. Prompt 不是后处理，而是直接进入 backbone token
- 文件: `torchreid/models/promptable_transformer_backbone.py`
- 核心函数: `_mask_embed()`
- 做法:
  - 将 keypoint prompt 预处理成 dense prompt masks
  - 再把 prompt token / prompt patch embedding 直接加到 image tokens 上
  - 推理时若禁用 prompt，会喂一个固定“空 prompt”，保证训练/测试分布一致
- 含义: KPR 不是把 prompt 当辅助标签，而是把 prompt 当输入条件

### 2. 模型始终保留多组结构化 embedding
- 文件: `torchreid/models/kpr.py`
- 输出不是单一 embedding，而是同时维护：
  - `global`
  - `foreground`
  - `background`
  - `concat_parts`
  - `parts`
- 每个 part 还有对应 `visibility_scores`
- 这意味着结构化局部信息会一直保留到检索阶段，而不是先压成一个向量

### 3. 训练目标核心不是新 classifier，而是 GiLt + visibility-aware supervision
- 文件: `torchreid/engine/image/part_based_engine.py`
- 文件: `torchreid/losses/GiLt_loss.py`
- 训练时：
  - holistic 分支主要吃 ID loss
  - part 分支主要吃 local triplet
  - visibility 既可以二值筛选，也可以连续加权
- 含义: KPR 真正依赖的是“结构化 part 表征 + 可见性感知训练”，不是小模块堆叠

### 4. 检索阶段直接做 mutually-visible part matching
- 文件: `torchreid/metrics/distance.py`
- 核心函数: `compute_distance_matrix_using_bp_features()`
- 做法:
  - 若 visibility 是二值，query / gallery 只在共同可见 part 上取均值距离
  - 若 visibility 是连续值，使用 `sqrt(v_q * v_g)` 作为 pair-specific 权重
- 关键点: **visibility 主要作用在 pairwise distance 上，而不是只在训练时改 pooling 权重**

## 对当前代码线的直接启发
1. `exp033 / exp034` 已经在数据层面处理了 target assignment，但我们在检索时仍然把 branch 压成一个向量，**没有做 pair-specific common-support reasoning**。
2. `exp030a` 的 GCN/KPP branch 已经证明价值主要发生在 fusion，而不是 global 提升；这说明 branch 的结构信息应该在检索阶段继续保留，而不是只拿来拼接。
3. KPR 的真正创新点是“目标歧义 + 共同可见 part 检索”，不是 learnable weighting 本身。

## 对我们的约束
- 不能再把下面这些点当主创新：
  1. `visibility weighting`
  2. `learnable keypoint / part weighting`
  3. “支持 prompt / target-aware” 这种泛化说法
- 因为 KPR 已经把这些概念清楚地定义并落到完整代码路径里了。

## 我们还能争什么
1. **不用 parsing part，而是直接用 keypoint/skeleton branch 做共同可见匹配**
2. **把 target-aware pose branch 的结构信息延续到 retrieval-time reasoning**
3. **在不引入 prompt 的默认设定下，先用当前 branch 验证 pair-specific keypoint matching 是否已经成立**

## 当前判断
- KPR 说明：真正值得推进的是 **target ambiguity / common visible support / retrieval-time reasoning**。
- 这直接削弱了把 `AFF` 或新的内部加权 MLP 当主线的合理性。
