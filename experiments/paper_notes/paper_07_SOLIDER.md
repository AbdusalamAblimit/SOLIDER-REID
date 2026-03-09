# Paper 7: SOLIDER - Semantic-Aware Learning for Human Representation
**来源**: CVPR 2023
**仓库**: https://github.com/tinyvision/SOLIDER
**核心**: 语义-外观解耦的自监督预训练

## 核心机制: semantic_weight

### 构造
- `semantic_weight` 是 [p, 1-p] 二维向量, p=语义比例
- 训练时: Teacher(全语义) → 伪标签; Student(随机权重) → 学习解耦

### 在 Swin 中的应用 (每个 stage)
```python
sw = semantic_embed_w(semantic_weight)  # Linear(2 → feat_dim)
sb = semantic_embed_b(semantic_weight)  # Linear(2 → feat_dim)
x = x * softplus(sw) + sb  # 仿射变换
```

### 关键洞察
1. SOLIDER 的语义调制(sw, sb)已内嵌在我们的预训练权重中
2. **创新方向**: 将 semantic_weight 从全局二元扩展为"Pose-Aware Semantic Control"
   - 对不同 body part 应用不同的语义权重
   - 遮挡区域降低语义权重(不可靠), 可见区域保持/提高
3. 我们框架的独特优势: SOLIDER预训练 + 姿态信息 = 尚未被探索的组合
