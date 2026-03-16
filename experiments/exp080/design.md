# 实验 exp080: Adaptive Test-Time Fusion (ATF)

## 动机
- subset analysis: PAA equal_concat 在多人图 +1.69%/+2.02%，单人图 +0.47%/-1.61%(R1)
- 单人图不需要 GCN branch 的互补信息，反而被噪声干扰
- **假设**: 对不同 query 使用不同的 test-time fusion 可以同时获得两类图的最优性能

## 技术方案 (test-time only, 不需要训练)
- 使用 exp066 PAA 的已有 checkpoint
- query 根据 num_persons 自适应选择:
  - 单人图: global-only 特征
  - 多人图: equal_concat 特征
- gallery 固定使用 equal_concat（因为 gallery 的 pose 信息在检索时未知）

## 注意
- gallery 全部使用 equal_concat 意味着 global-only query 和 equal_concat gallery 之间维度不匹配
- 需要让 global query 也 padding 到 equal_concat 的维度，或只用 global 部分计算距离
- 最简方案: query 中单人图使用 global 部分 zero-pad 到 full concat dim

## 对照
- exp066 PAA equal_concat = 61.6%/74.2%

## 实现
- 纯评估脚本，不需要训练
