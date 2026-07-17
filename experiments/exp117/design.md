# 实验 exp117: Visibility-Conditioned Graph Attention (VCGA)

## 动机

SCKD 系列（exp110-116, 7 个变体）已经证明：EMA prototype bank 的增量上限仅 ~+0.1% mAP。这说明从外部（memory bank）向模型注入 support-complete 信号行不通。

但 oracle experiment（exp109, +8.5% mAP）证明 headroom 真实存在。差距可能不在"更好的 teacher"，而在**模型内部的信息传递效率**。

当前 GCN 使用固定的 COCO skeleton adjacency matrix（对称归一化），不区分可见和遮挡 keypoint：
- 可见 keypoint → 遮挡 keypoint 和 遮挡 → 可见 的权重完全相同
- 但理想情况下，遮挡 keypoint 应该更多地从可见邻居"接收"信息
- 可见 keypoint 不应被遮挡邻居的噪声"污染"

## 核心假设

如果 GCN 的消息传递权重由 keypoint 可见度调制，使得：
- 高可见度 keypoint 作为强信号源（发送更多信息）
- 低可见度 keypoint 作为弱信号源（发送更少信息）

则 GCN 能更有效地从可见区域向遮挡区域传播身份信息，提升最终的 pooled feature 质量。

## 技术方案

修改 `SkeletonGCN.forward()`，加入 visibility-conditioned attention：

```python
def forward(self, x, kp_weights=None):
    """
    Args:
        x: (B, 17, C) keypoint features
        kp_weights: (B, 17) confidence scores (optional)
    """
    if kp_weights is not None:
        # Scale adjacency by source visibility:
        # neighbor j's contribution to node i is scaled by vis(j)
        vis = kp_weights.unsqueeze(1)  # (B, 1, 17) = source visibility
        adj = self.adj_norm.unsqueeze(0) * vis  # (B, 17, 17)
        # Re-normalize rows to sum to 1
        row_sum = adj.sum(dim=2, keepdim=True).clamp(min=1e-6)
        adj = adj / row_sum
    else:
        adj = self.adj_norm

    h = x
    for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
        h = torch.matmul(adj, h)
        h = layer(h)
        h = norm(h)
        if i < len(self.layers) - 1:
            h = F.relu(h, inplace=True)

    return x + h
```

关键改动：
- `adj_norm * vis(j)` 使得高可见度 keypoint 的信号权重更大
- 行级重归一化保持 aggregation 的稳定性
- 当 `kp_weights=None` 时退化为标准 GCN（兼容已有实验）

## 对照组

1. 主对照: `exp030a-eq seed1234`（标准 GCN）
2. 确认基线一致性：warmup 阶段应与 exp030a 完全一致

## 预期结果

如果假设成立：
1. 遮挡 query 上改善最明显（低可见度 keypoint 获得更好的恢复）
2. 非遮挡 query 上中性（所有 keypoint 可见时 adj ≈ 原始 adj_norm）
3. 预期 mAP +0.3~0.5%（基于 GCN 改善的经验幅度）

如果失败，最可能原因：
1. 可见度分数本身不够准确，引入噪声
2. GCN 只有 2 层，信息传播范围有限，调制效果不明显
3. 当前 GCN 的 zero-init residual 已经很好地处理了遮挡情况

## 风险与失败解释

1. 低风险：改动极小（10 行代码），不影响已有实验
2. 若失败，说明 GCN 内部路由不是瓶颈
3. 后续可以尝试更复杂的 GAT 变体
