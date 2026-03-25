# 实验 exp189: Visibility-Weighted SupCon (Structural Contrastive 简化版)

## 动机
- 当前 SupCon 对所有 6 个 structural tokens 等权计算
- 但在 PLBOA 下，部分 tokens 来自被遮挡区域（特征质量低）
- 如果用 pose visibility 加权 SupCon（高可见 token 权重大，低可见 token 权重小），
  可以让 contrastive 更关注高质量 token → 更好的 metric space

## 核心假设
Visibility-weighted per-token SupCon 比 uniform per-token SupCon 更好，
因为它让高质量（可见）的 body-part tokens 在 contrastive learning 中获得更大权重。

## 技术方案
- 修改 make_loss.py: 在 SupCon path 中，计算每个 token 的 visibility weight
- Weight = mean heatmap response for that token's body-part group
- SupCon loss = weighted average of per-token SupCon losses

## 对照组
- exp176 (uniform SupCon): 64.1/75.5
- exp179 (SupCon base): 64.2/74.9
