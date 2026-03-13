# 实验 exp039: 共同可见关键点检索诊断

## 动机
- `exp030a` 多种子已经确认：GCN/KPP branch 的价值主要体现在 fusion，而不是 global 提升。
- `exp035b / exp036 / exp037` 这一轮 branch 内部调权重、调 loss 的实验收益很弱，说明问题可能不在 branch 内部，而在 **branch 信息如何进入检索距离**。
- 文献/代码复盘（KPR, BPBreID, FRT, QPM）显示：近年的 occluded ReID 更强调 **query-gallery 共同可见支撑**，而不是单图自适应加权。
- 当前 `equal_concat` 会在距离计算前把 GCN branch 压成单一向量，浪费了关键点级结构信息。

## 核心假设
- 如果 GCN branch 确实学到了语义对齐的关键点级局部表征，那么在测试阶段只比较 query/gallery **共同可靠** 的关键点距离，应优于直接做固定 `equal_concat`。
- 即使 `cvk_only` 不一定超过 `equal_concat`，`global + common-visible keypoint distance` 也可能更接近 branch 的真实使用方式。

## 技术方案

### 新增测试模式
在 `MODEL.POSE_TEST_FEAT` 中新增：
- `cvk_only`: 只使用共同可见关键点距离
- `cvk_hybrid`: `global distance` 与 `common-visible keypoint distance` 做距离级融合

### 数据流
1. backbone 正常输出 `global_feat`
2. skeleton head 在测试时保留：
   - `kp_feats`: GCN 增强后的 17 个关键点特征
   - `kp_weights`: 当前 keypoint reliability（默认沿用 `score`）
3. evaluator 计算：
   - `dist_global(q, g)`
   - 每个关键点的 `dist_k(q, g)`
   - `w_k(q, g) = sqrt(w_q^k * w_g^k)`
   - `dist_cvk(q, g) = weighted_mean_k(dist_k, w_k)`
4. 输出：
   - `cvk_only`: `dist = dist_cvk`
   - `cvk_hybrid`: `dist = (dist_global + dist_cvk) / 2`

## 对照组
- 主基线结论：`exp030a-eq` 3-seed mean = `60.73% mAP / 72.57% R1`
- 当前代码线可直接评测权重：`exp035a` checkpoint = `61.1% mAP / 73.8% R1`
- 本实验先做两个诊断子实验：
  - `039a`: `cvk_only`
  - `039b`: `cvk_hybrid`

## 预期结果
- 理想结果：
  - `cvk_hybrid >= equal_concat`
  - 说明 branch 的局部结构信息更适合在距离级推理中使用
- 次优结果：
  - `cvk_only < equal_concat` 但 `cvk_hybrid` 接近或略超 baseline
  - 说明 global 仍是主体，keypoint branch 更适合作为 pair-specific 补充
- 负结果：
  - `cvk_only` 和 `cvk_hybrid` 都不如 `equal_concat`
  - 说明当前 branch 虽然对 concat 有帮助，但关键点级局部距离本身还不够稳

## 风险与失败解释
1. 当前 `kp_weights` 默认来自 `score`，它不是严格的可见性标签，可能导致共同可见估计不准。
2. 单个关键点特征比 parsing part 更稀疏，pairwise 距离可能噪声更大。
3. 这是 retrieval-time diagnostic，不应把增益表述成训练端创新。
4. 若结果为负，也有价值：
   - 可以说明 branch 的增益主要来自 embedding-level complement，而不是可直接解开的 keypoint-level pairwise matching。
