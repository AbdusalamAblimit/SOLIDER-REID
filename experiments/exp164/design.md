# 实验 exp164: STD-PR V2 — Anchor-Sampled Queries

## 改进
STD-PR V1 用 learnable queries 初始化 → R1 低 6.3（缺空间精度）。
V2 用 keypoint 坐标处的 bilinear-sampled features 初始化 queries。
Cross-attention 变成 refinement（而非 from-scratch aggregation）。

## 技术改动
- Query 初始化：learnable → bilinear_sample(feat_map, part_centroids) + learnable_embedding
- Cross-attention 变成残差 refinement（已有 residual connection）
- Zero-init attention output projection 让初始行为 = 纯采样（暂不做，先看效果）

## 预期
- R1 应该接近 GCN（因为空间精度保留了）
- mAP 可能比 V1 更好（anchor + context refinement > pure attention）
