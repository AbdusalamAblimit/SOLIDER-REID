# 实验 exp089: Pose-Aware Matching Network (PAMN)

## 动机 — 83 个实验的根本教训

83 个实验的核心教训：
1. 对 backbone feature 的所有后处理修改都无效（PGFI、TDPC、APG、CIPGFR）
2. 所有辅助 loss 都无效（MM、SGMKC、CSGT、PAMC、PAML、KDL）
3. PSG+PAA 已把 feature extraction 推到极限
4. **未触及的环节：matching/retrieval 方式**

当前 matching：concat → L2 distance → ranking。完全忽略 pose 结构对齐。
CVK 已证明 keypoint matching 有 +0.8% mAP 正信号，但只是简单加权 L2。

## 创新点
**学习 pose-aware matching**，而非更好的 feature。

训练一个 Matching Network：
- 输入：query 和 gallery 的 17 个 keypoint features + visibility
- 输出：matching score
- 学习：哪些 keypoint 在当前 pair 上最判别，如何处理部分遮挡

## 与现有方法的区别
- vs CVK: 手工加权 vs learned matching
- vs PAML: PAML 改 triplet distance（失败了），PAMN 是独立 matching 模块
- vs Re-ranking: post-processing vs end-to-end learnable

## 技术方案

### Matching Network
```python
class PoseMatchingNetwork(nn.Module):
    # Input: kp_feats_q (17, 768), kp_feats_g (17, 768), vis_q (17,), vis_g (17,)
    # Output: score (scalar)

    def forward(self, kp_q, kp_g, vis_q, vis_g):
        # 1. Per-keypoint cosine similarity
        sim = F.cosine_similarity(kp_q, kp_g, dim=-1)  # (17,)
        # 2. Common visibility mask
        mask = vis_q * vis_g  # (17,)
        # 3. Masked similarity aggregation
        masked_sim = (sim * mask).sum() / mask.sum().clamp(min=1)
        # 4. Cross-keypoint interaction (optional: which kp pairs discriminate?)
        diff = (kp_q - kp_g).pow(2).sum(-1)  # (17,) per-kp distance
        # 5. Learned scoring from [sim, mask, diff]
        features = torch.cat([sim, mask, diff], dim=-1)  # (51,)
        score = self.mlp(features)  # scalar
        return score
```

### 训练
- 在 batch 内构造 pairs (pos + neg)
- 对每个 pair 计算 PAMN score
- Loss: contrastive loss (pos score high, neg score low)
- 与 backbone 联合训练（但 kp_feats 从 detached GCN output 获取，避免干扰）

### 测试
- 阶段 1: global+GCN concat → L2 粗排序 → top-100
- 阶段 2: PAMN re-score top-100 → 精排序

## 参数
- MLP(51 → 128 → 1): ~7K params

## 对照
- exp066 PAA = 61.6%/74.2%
- CVK hybrid = 61.9%/73.2%

## 为什么这是真正的创新
1. **问题转换**: extraction → matching
2. **机制新颖**: learned keypoint-pair matching（不是简单距离）
3. **证据清晰**: vs CVK、vs L2、vs equal_concat 直接对比
