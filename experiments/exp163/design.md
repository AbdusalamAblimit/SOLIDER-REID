# 实验 exp163: PNIS (Pose-Normalized Identity Space)

## 范式级创新

### 问题
同一个人在不同 pose 下的 feature 差异很大。正面 vs 背面、站 vs 坐、全身 vs 半身——这些 pose variation 是 feature space 里的主要噪声。

### 核心假设
如果我们能显式地把 pose-induced variation 从 feature 里"减掉"，剩下的就是更纯的 identity information。

### 公式
```
raw_feat = backbone(image)              # 包含 identity + pose 信息
pose_offset = PoseEncoder(skeleton)     # 纯 pose 信息
identity_feat = raw_feat - pose_offset  # factor out pose
```

匹配时用 `identity_feat` 而非 `raw_feat`。

### 为什么让人眼前一亮
- 这像 "style transfer" 里的 content-style 分离
- Pose 是 "style"（同一个人的不同表现），identity 是 "content"
- 减法操作极简但意义深刻

## 技术方案

### PoseEncoder
- 输入：17 个 keypoint 的 (x, y, score) = 51 维
- 网络：MLP(51 → 256 → 768)
- 输出：768-d pose offset vector
- 训练时用 L2 loss 让 PoseEncoder 预测 pose 信息
- 同时 identity_feat 做 ID classification（如果 pose 被减掉了，identity_feat 应该更纯）

### 辅助 loss
- Pose prediction: 从 raw_feat 预测 skeleton → 确认 raw_feat 确实包含 pose 信息
- Identity classification: 在 identity_feat 上 → 确认减法后的 feature 仍有判别力

### 与 PVAT 的区别
- PVAT: adversarial gradient reversal（间接、弱信号）→ 失败
- PNIS: 显式减法（直接、强操作）→ 更有可能生效

### 与 PSG 的关系
- PSG 在 backbone 内部用 pose 调制特征（process 阶段）
- PNIS 在 backbone 之后减掉 pose（output 阶段）
- 两者正交

## 实现

### 修改 GCN branch 的 pooled feature
```python
# 原来：gcn_feat = confidence_weighted_pool(kp_feats)
# 现在：
gcn_feat = confidence_weighted_pool(kp_feats)
pose_input = torch.cat([keypoints.flatten(), scores], dim=-1)  # (B, 51)
pose_offset = self.pose_encoder(pose_input)  # (B, 768)
identity_feat = gcn_feat - pose_offset  # factor out pose
```

### 训练 loss
- Global ID + triplet: 不变（不动 global branch）
- GCN ID: 在 `identity_feat` 上（不是 `gcn_feat`）
- GCN triplet: 在 `identity_feat` 上
- Pose prediction auxiliary: MSE(raw_feat 预测的 skeleton, 真实 skeleton)

### Test-time
- 用 `identity_feat` 做 equal_concat matching
- Pose offset 在 test-time 也计算（需要 pose 信息，我们有）

## 预期
- 如果 pose variation 确实是 feature space 的主要噪声 → identity_feat 更纯 → mAP 提升
- 如果 pose 信息对 identity 也有帮助（如 gait, body shape）→ 减太多会伤害 → 可以用 learnable alpha: `identity_feat = raw_feat - alpha * pose_offset`

## 设计决策：CE/Triplet 梯度不对称

GCN head 在返回 `gcn_feats` 之前已经计算了 `cls_score`（在 raw feature 上）。PNIS 在返回之后替换 `gcn_feats[0]`。因此：
- **CE loss** 训练 GCN 的 raw feature（不经过 PNIS）
- **Triplet loss** 训练 PNIS 的 identity_feat（经过减法）

这意味着 PNIS 只从 triplet loss 获得训练信号。这是有意的：
- CE loss 确保 GCN 产出高质量的 raw feature
- Triplet loss 确保减法后的 identity_feat 有更好的 metric learning 性质
- 两者不冲突因为 CE 训练的是 classifier weight，triplet 训练的是 feature space geometry

## alpha 初始化

alpha = sigmoid(-3.0) = 0.047。初始只减掉 5% 的 pose offset，随训练逐渐增大。

## 风险
1. pose_offset 可能无法准确捕捉 pose variation（51 维太少）
2. 减法可能过度或不足
3. 15K 数据可能不够训练 PoseEncoder
4. 仅 triplet loss 训练 PNIS 可能信号不够
