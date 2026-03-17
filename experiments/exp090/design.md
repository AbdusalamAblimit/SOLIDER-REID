# 实验 exp090: Skeleton-Guided Cross-Image Feature Recovery (SGCFR)

## 动机 — 范式级创新

当前所有方法（包括我们的 83 个实验）都在做：
  单图 → 特征提取 → 单向量比较

但遮挡问题的根本是：**单张图信息不够**。

**SGCFR 的核心洞察**: 在 gallery 中可能存在同一人的其他图片。
如果 query 的下半身被遮挡，但 gallery 中有同一人站着的全身照，
我们可以用那张图的下半身 keypoint features 来"填补" query 的缺失部分。

## 创新点

1. **问题层面**: 从"单图特征"到"跨图协作特征恢复"
2. **机制层面**: 用 skeleton structure 对齐 keypoint features，只恢复遮挡部分
3. **证据层面**: vs NFC (global level) / vs CVK (pairwise) / vs equal_concat (no recovery)

## 与现有方法的区别

- vs NFC: NFC 在 global 空间做 neighbor centralization（不考虑 pose structure）
- vs FRT: FRT 用 transformer decoder 做 feature recovery（需要训练，复杂）
- vs Re-ranking: Re-ranking 不修改特征（只重排序）
- **SGCFR**: 用 skeleton structure 在 keypoint 空间做 cross-image completion（不需要训练！）

## 技术方案（Test-time only，不需要训练）

```
输入: 已有 checkpoint 的 per-query kp_feats (17, 768) + kp_weights (17,)

Step 1: 粗检索
  - 用 equal_concat 做 L2 检索 → top-K 候选

Step 2: 对每个 query, 从 top-K 候选中恢复遮挡特征
  For each query:
    kp_q (17, 768), vis_q (17,)
    
    # 找到 top-K 中与 query 最可能是同一人的候选
    candidates = top_K_gallery
    
    For each keypoint k where vis_q[k] < threshold:
      # query 的这个关键点被遮挡
      # 从候选中找到这个关键点可见的图片
      visible_cands = [c for c in candidates if vis_c[k] > threshold]
      if visible_cands:
        # 用候选的可见 keypoint feature 恢复 query 的遮挡 feature
        recovered_feat[k] = weighted_avg([c.kp_feat[k] for c in visible_cands])
        vis_q_recovered[k] = 1.0
    
    # 用恢复后的 keypoint features 重新计算 GCN branch feature
    # 或直接用恢复后的 kp_feats 做 CVK matching

Step 3: 用恢复后的特征重新排序
```

## 实现
- 只需要修改 test.py 或写独立评估脚本
- 使用 exp066 PAA 的已有 checkpoint
- 不改模型，不重训练

## 参数
- 零额外参数（test-time only）

## 关键设计选择
1. K = 多少个候选？（5, 10, 20）
2. 遮挡阈值？（0.3, 0.5）
3. 恢复策略？（最近邻 / 加权平均 / 只用 top-1）

## 对照
- exp066 PAA equal_concat = 61.6%/74.2%
- exp040b CVK hybrid = 61.9%/73.2%
- exp049 NFC k=5 = 67.3%/77.6%

## 为什么这是真正的创新
- 不是"加一个模块/loss"
- 而是"改变如何使用已有信息"
- 利用 gallery 中的冗余信息来恢复 query 的遮挡部分
- 和 Pose2ID 的 "feature centralization" 是平行的思路，但用 skeleton structure 而非生成模型
