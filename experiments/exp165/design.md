# 实验 exp165: STD-PR + Confidence-Gated Pooling

## 动机
STD-PR V1 的 R1 低（67.4 vs GCN 73.7）。原因之一：6 tokens mean pooling 对遮挡 token 不降权。

## 改动
在 mean pooling 前加 per-token reliability weighting：
- 从 pose heatmap response 获取 per-part visibility score
- 用 score 对 6 tokens 加权平均
- 遮挡的 body-part token 自动降权

## 实现
在 model forward 中，从 scene_heatmaps 计算 per-part 平均响应：
```python
# 6 parts 对应的 heatmap channels
part_groups = [[0,1,2,3,4], [5,6,11,12], [5,7,9], [6,8,10], [11,13,15], [12,14,16]]
part_weights = []
for g in part_groups:
    part_weights.append(hm[:, g].mean(dim=(1,2,3)))  # (B,) per-part average response
part_weights = torch.stack(part_weights, dim=1)  # (B, 6)
part_weights = part_weights / part_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

str_feat = (structural_tokens * part_weights.unsqueeze(2)).sum(dim=1)  # weighted average
```

## 预期
- R1 应改善（遮挡 token 被降权）
- mAP 可能持平或微正
