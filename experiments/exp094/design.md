# 实验 exp094: Pose-Conditional Query Adaptation (PCQA)

## 根本洞察
当前 ReID: 每个 query 生成一个固定 feature，去和所有 gallery 比较。
但 query 和不同 gallery 的可比较性不同——当两个人 pose 不同时，应该"翻译"到同一 pose 下再比较。

## 核心创新
训练一个 **Pose Translation Module (PTM)**: 
- 输入: query_global_feat (768) + query_pose (17d) + gallery_pose (17d)
- 输出: translated_query_feat (768) — "如果 query 的人以 gallery 的 pose 出现，feature 会是什么"

训练监督: 同一 ID 的不同 pose 图片对
- (img_A, pose_A) 和 (img_B, pose_B) 是同一人
- PTM(feat_A, pose_A, pose_B) ≈ feat_B
- PTM(feat_B, pose_B, pose_A) ≈ feat_A

测试: 对每个 query-gallery pair，先用 PTM 翻译 query feat 到 gallery 的 pose，然后比较翻译后的距离。

## 这和谁不同？
- vs Pose2ID: Pose2ID 生成图片再提取特征（需要 diffusion）。我们直接在 feature space 翻译。
- vs KPR: KPR 用 keypoint prompt 做 part-level retrieval。我们做 pose-conditional feature translation。
- vs SGCFR: SGCFR 从 gallery 恢复 query 遮挡部分。PCQA 把 query 翻译到 gallery 的 pose。

## 技术方案
```python
class PoseTranslationModule(nn.Module):
    # Input: src_feat (768), src_pose (17), tgt_pose (17)
    # Output: translated_feat (768)
    
    def __init__(self):
        self.pose_encoder = MLP(17*2, 256, 256)  # concat src+tgt pose
        self.translator = nn.Sequential(
            nn.Linear(768+256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 768),
        )
        # Zero-init: starts as identity (output = src_feat)
        nn.init.zeros_(self.translator[-1].weight)
        nn.init.zeros_(self.translator[-1].bias)
    
    def forward(self, src_feat, src_pose, tgt_pose):
        pose_cond = self.pose_encoder(cat(src_pose, tgt_pose))
        delta = self.translator(cat(src_feat, pose_cond))
        return src_feat + delta  # residual translation
```

## 训练
- 在 batch 内，对每个同 ID pair (A, B):
  - loss += MSE(PTM(feat_A, pose_A, pose_B), feat_B.detach())
  - loss += MSE(PTM(feat_B, pose_B, pose_A), feat_A.detach())
- PTM 的参数通过这个 loss 训练
- 主 loss (ID + Triplet) 不变

## 测试
- 方式 1 (快): 对 top-K gallery candidates，用 PTM 翻译 query，重新计算距离
- 方式 2 (慢但更准): 对所有 gallery，用 PTM 翻译后比较

## 参数
- pose_encoder: ~70K
- translator: ~800K
- 总: ~870K

## 为什么这是真正的创新
1. **问题新**: 不是"如何提取更好的特征"，而是"如何让不同 pose 的特征可比"
2. **机制新**: feature-space pose translation（不需要生成模型）
3. **证据**: 可以展示翻译前后的距离变化、可视化翻译效果
