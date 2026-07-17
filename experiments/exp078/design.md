# 实验 exp078: Adaptive PAA Gating (APG)

## 动机
- exp066 subset analysis 发现 PAA 增益全部来自多人图 (+1.69%/+2.02%)
- **单人图上 PAA R1 退化 -1.61%** — PAA 在无遮挡场景是有害的
- 当前 PAA 对所有图片一视同仁，无论是否有遮挡
- 如果能让 PAA 只在多人/遮挡场景生效，应该能同时：
  - 保留多人图的 +1.69% 增益
  - 消除单人图的 -1.61% 退化
  - 预期净增益 > 当前 PAA 的 +1.07%

## 创新点 / 核心想法
- **核心假设**: PAA 的 pose adapter 在多人遮挡场景有效，但在单人清晰场景有害。一个自适应 gate 可以解决这个问题。
- **gate 信号来源**: heatmap 本身。多人图的 scene heatmap 热量更分散（多个人的关节点分布更广），单人图的热量更集中。
- **gate 设计**: 极简 — 1 个标量 gate，从 scene heatmap 的全局特征预测

## 技术方案
```
scene_heatmaps (B, 17, H, W)
    ↓
GAP → (B, 17) → MLP(17→1) → sigmoid → gate (B, 1)
    ↓
PAA output *= gate  →  x = x + gate * adapter(heatmap)
```

对于单人图：heatmap 集中 → gate 学到低值 → PAA 被抑制
对于多人图：heatmap 分散 → gate 学到高值 → PAA 正常工作

### 参数增量
- MLP: 17 → 1 = 17+1 = 18 参数
- 实际上几乎零参数增加

### 修改位置
- `model/modules/pose_additive_adapter.py`: PoseAdditiveAdapter.forward() 中添加 gate
- 或在 `pose_backbone_model.py` 的 PAA 调用处添加 gate

### 初始化
- MLP bias 初始化为 0，weight 初始化为 0 → sigmoid(0) = 0.5 → 初始 gate = 0.5
- 这意味着开始时 PAA 贡献减半，通过训练学到自适应

## 预期结果
- 如果成功: 多人图保持 +1.69%，单人图退化被消除或减小 → 整体 mAP > 61.6%
- 如果失败: gate 学不到有意义的分配，退化为 uniform scaling

## 对照组
- **Baseline**: exp066 PAA = 61.6%/74.2%
- **消融变量**: 仅在 PAA 输出上加 gate (1个标量 MLP)

## 风险
- gate 可能直接学到 ~0.5 的常数（等效于把 PAA adapter 的 scale 减半）
- 17 维 heatmap summary 可能不足以区分单人/多人
