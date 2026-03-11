# 实验 exp027: PCRA (Pose-Contrastive Representation Alignment)

## 动机
- 26 个实验证明所有 forward path 添加/修改都会干扰 PSG 的梯度（exp008-021）
- exp026 (SPD) 证明 pose 信号在 Occluded-Duke 上一致有用，不存在过度依赖
- 红蓝队辩论中 PCRA 获胜（8/10 vs 7/10），理由：操作在从未探索的维度（loss 距离度量）
- 当前 triplet loss 在 hard example mining 时完全不感知两个样本的 pose 相似性——一个全身人和一个只露上半身的人被视为等价的 negative pair

## 创新点 / 核心想法
**核心假设：在 hard example mining 中考虑 pose 相似性，优先选择 pose 相似但 ID 不同的样本作为 hard negative，可以提升判别力。**

具体地：
1. GAP 每个样本的 17 通道 scene heatmap → 17 维 pose signature
2. 计算批内所有样本对的 pose cosine similarity → (N, N) 矩阵
3. 用 pose similarity 调制 distance matrix：`adjusted_dist = dist_mat * (1 - alpha * pose_sim)`
4. 在 adjusted_dist 上做 hard example mining（选择哪些对）
5. 用 adjusted_dist 计算 triplet loss

效果：
- Pose 相似的 negative pair：距离被缩小 → 更可能被选为 hard negative → 模型被迫区分相同姿态但不同身份的人
- Pose 不同的 negative pair：距离几乎不变 → 不太可能被选为 hard negative → 避免对遮挡差异的无意义惩罚

## 技术方案

### 修改文件
1. **`config/defaults.py`**: 新增 `POSE_PCRA_ALPHA = 0.0`（默认关闭）
2. **`loss/triplet_loss.py`**:
   - `TripletLoss.__init__` 接收 `pose_alpha` 参数
   - `TripletLoss.__call__` 接收可选 `pose_sim` 矩阵
   - 用 `pose_sim` 调制 `dist_mat`
3. **`loss/make_loss.py`**:
   - 构造 `TripletLoss` 时传入 `pose_alpha`
   - `loss_func` 接收可选 `pose_sim` 参数
4. **`processor/processor.py`**:
   - 训练时如有 pose_dict，计算 pose signature 和 pose_sim
   - 传给 loss_func

### 数据流
```
Pose heatmap (B, 17, H, W)
    → GAP → (B, 17) pose signature
    → cosine_similarity(sigs, sigs) → (B, B) pose_sim matrix
    → 传给 triplet loss
    → adjusted_dist = dist_mat * (1 - alpha * pose_sim)
    → hard mining on adjusted_dist
    → soft margin loss
```

### 关键超参数
- `POSE_PCRA_ALPHA = 0.2`（初始值，控制 pose 调制强度）
- 其余所有参数与 exp007 (PSG) 完全相同

## 预期结果
- **如果假设成立**: mAP > 58.3%（超过 PSG），证明 pose-aware mining 有效
- **如果中性 ≈ 58.3%**: pose similarity 没有提供额外的 mining 信号
- **如果 < 58.3%**: pose similarity 扰乱了 hard mining 的质量
- **最可能失败原因**: 17 维 pose signature 的余弦相似度不够区分"相同可见部位"和"不同可见部位"

## 对照组
- **直接对照**: exp007 PSG (mAP 58.3%, R1 67.9%)
- **消融变量**: 本实验仅修改 triplet loss 的距离度量，不改变模型架构或 forward pass
