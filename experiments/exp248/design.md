# exp248: PCFD — Pose-Conditioned Feature Differencing

## 核心创新 (范式级)

**重新定义 ReID 检索问题**:
- 传统: feature → cosine distance → rank
- PCFD: feature pair → per-part difference → same/diff classification → rank

"不问两人有多像, 而问在共同可见部位上有什么不同"

## 技术方案

### Phase 1: Test-time 验证 (本轮)

在已有 checkpoint (exp191/exp244) 上做 test-time PCFD:

1. 提取 query/gallery 的 per-part features (已有: LGPA-D 或 GCN keypoint features)
2. 对 top-K candidates, 计算 per-part feature 差异
3. 用轻量 MLP 分类 same/diff ID
4. MLP 在 training set pairs 上学习 (不需要重新训练 backbone)

### 架构

```
Query parts: [head, torso, arms, legs]  (LGPA-D 或 GCN features)
Gallery parts: [head, torso, arms, legs]

For each part k:
  diff_k = abs(query_k - gallery_k)  或  query_k * gallery_k (element-wise)
  vis_k = min(query_vis_k, gallery_vis_k)  # 共同可见性

Pose-conditioned differencing:
  weighted_diff = sum(vis_k * diff_k) / sum(vis_k)
  → MLP → same/diff score

Re-ranking: 用 same/diff score 调整初始 ranking
```

### 为什么这个可能有效

1. 传统 cosine 对所有维度等权 → PCFD 在遮挡部位上给 0 权重
2. 传统 cosine 是 holistic 比较 → PCFD 是 part-level 精细比较
3. MLP 可学习 "哪些 part 差异更 discriminative" 
4. 与 NFC/re-ranking 不同: PCFD 是 pose-conditioned 的, 理解身体结构

### 实现

1. `scripts/eval_pcfd.py` — test-time PCFD 评估脚本
2. 在 exp244 checkpoint 上验证 (per-part features 已有)
3. 不需要重新训练 backbone

## 对照组
- exp244 equal_concat (cosine): 65.3/75.7
- exp244 + NFC: 未测试
- exp244 + re-ranking: 未测试
- **exp244 + PCFD**: 目标

## 预期结果
- 成功: +1~3% mAP (per-part differencing 比 holistic cosine 更精准)
- 失败: ~0% (MLP 学不到有用的差异模式)

## 实际结果 — 失败

PCFD test-time re-ranking 全面负面:
- alpha=0.1: -13.2% mAP, -5.2% R1
- alpha=0.3: -18.5% mAP, -7.6% R1

MLP difference classifier 严重过拟合训练集 pairs, 完全不泛化到 test set。
简单 MaxSim (无学习, max cosine) 反而有效 (+0.7%)。

**结论**: learned pair-level matching 在 ReID 上不行, 不管是训练端 (MaxSim training exp152/153) 还是 test-time (PCFD MLP)。feature-level cosine / MaxSim 是更好的选择。
