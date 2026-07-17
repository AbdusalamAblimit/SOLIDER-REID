# exp156 SPLADE Claude 审查

## 第一轮：NOT PASSED (3 Critical/High)
1. Critical: sparse feature 在 equal_concat 中占 57% 维度 → 固定：test-time 不用 sparse
2. High: GCN CE/triplet 被稀释 50% → 固定：不再 append 到 gcn 列表
3. High: euclidean triplet on sparse features → 固定：SPLADE 只做 auxiliary CE

## 第二轮修复后：PASSED
- SPLADE 现在是纯训练辅助信号
- 训练：GCN feat → sparse projection → auxiliary CE (0.5 weight) + sparsity reg
- 测试：完全不变（global + GCN dense, same as exp030a）
- GCN 列表不受影响，loss 不被稀释
