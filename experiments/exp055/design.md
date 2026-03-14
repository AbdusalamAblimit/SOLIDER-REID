# 实验 exp055: PGAM 阈值消融 (threshold=0.5)

## 动机
- exp054 (PGAM, threshold=0.3) 显示正向信号: +0.37% mAP / +1.23% R1 vs 3-seed mean
- threshold=0.3 意味着 sigmoid(heatmap) > 0.3 的位置被视为 body（对应原始 logit > -0.85）
- 这是一个较宽松的阈值，大部分有任何人体响应的区域都被保留
- threshold=0.5（对应原始 logit > 0）更严格，只保留有明确人体响应的区域
- 需要验证更严格的阈值是否能进一步提升效果（更多 token 被 mask）

## 创新点 / 核心想法
- 本实验是 exp054 的消融变体，验证 PGAM 阈值敏感性
- 核心假设: 更严格的阈值 mask 掉更多"边缘"区域，可能进一步减少注意力污染

## 技术方案
- 与 exp054 完全相同，仅修改 `POSE_ATTN_MASK_THRESHOLD: 0.5`

## 预期结果
- 如果 threshold=0.5 > 0.3: 说明更严格的 masking 更好
- 如果 threshold=0.5 < 0.3: 说明 0.3 附近的"边缘"body 信息对注意力有价值
- 如果持平: 阈值不敏感，PGAM 鲁棒

## 对照组
- 对照: exp054 (threshold=0.3)
- 消融变量: 仅 threshold 从 0.3 → 0.5
