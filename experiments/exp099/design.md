# 实验 exp099: POT-Match (Partial Optimal Transport Matching)

## 动机
- 98 个实验的核心教训: 所有改进 features 的尝试（DPF, MRKF, PKP, 辅助损失）效果有限
- **真正的瓶颈不在 features，在 MATCHING**
- 当前匹配: 将每个人压缩成一个固定向量 → 余弦距离
- 问题: 固定向量丢失了遮挡结构信息（哪些部位可见、哪些被遮挡）
- **灵感**: Partial Optimal Transport (AAAI 2024) 提供了处理部分观测的数学框架

## 核心创新 — 范式级
**将 occluded ReID 匹配从「向量距离」升级为「最优传输」**

当前: person → feature vector → cosine distance
POT-Match: person → keypoint feature SET + visibility weights → Sinkhorn OT distance

关键优势:
1. 每个 (query, gallery) pair 的匹配策略自适应于双方的遮挡模式
2. 遮挡关键点自动获得低传输权重（不需要显式处理）
3. 可见部分的匹配是 OPTIMAL 的（最优传输找到最佳对应）
4. 完全可微分（Sinkhorn 算法），可用于训练

## 技术方案

### 匹配范式改变
```
旧: feat_q (768-d) × feat_g (768-d) → cosine similarity → rank

新: kp_feats_q (17, D) + weights_q (17)
    kp_feats_g (17, D) + weights_g (17)
         ↓
    Cost matrix: C[i,j] = 1 - cos(kp_q_i, kp_g_j)  (17×17)
         ↓
    Sinkhorn OT: T* = argmin <T, C> s.t. row/col marginals = weights
         ↓
    Distance = <T*, C>  (optimal transport cost)
         ↓
    Rank by distance
```

### 训练修改
- 保持现有 ID loss 和 pooled triplet loss（确保全局特征不退化）
- 新增: OT-based triplet loss 在 per-keypoint 特征上
  - Mining: 用 pooled 特征的 L2 距离找 hardest positive/negative
  - Loss: 用 Sinkhorn distance 计算 triplet loss
  - Weight: 可调，如 0.5

### 测试修改
- 新的评估脚本: `scripts/eval_pot.py`
- 计算 query-gallery pairwise Sinkhorn distance
- 与 global cosine distance 加权融合

### 实现
1. 新模块: `model/modules/sinkhorn_distance.py`
   - SinkhornDistance: log-domain Sinkhorn for numerical stability
   - OT-based triplet loss wrapper

2. `processor/processor.py`: 添加 OT triplet loss 分支

3. 评估脚本: `scripts/eval_pot.py`

### 效率分析
- Training: O(B × B × K² × N_iter) per batch = 64 × 64 × 289 × 20 ≈ 24M ops (negligible vs backbone)
- Testing: O(N_q × N_g × K² × N_iter) = 2K × 17K × 289 × 20 ≈ 197B ops → ~20s on GPU (acceptable)

### 关键超参数
- eps = 0.1 (Sinkhorn regularization, 控制 transport plan 的平滑度)
- max_iter = 20 (Sinkhorn 迭代次数)
- ot_weight = 0.5 (OT triplet loss 权重)

## 预期结果
- 如果成功: test-time OT matching 应该显著优于 cosine matching (+1~3%)
- 训练时 OT triplet 应该教会 per-keypoint 特征更好的辨别力
- 与 SGCFR 互补: SGCFR 恢复遮挡特征 → OT 用恢复后的特征匹配

## 对照组
- exp066 (PAA baseline, cosine matching): 61.6%/74.2%
- 消融:
  - OT matching only (无 OT training loss) — 测试 OT 匹配的独立价值
  - OT training + OT matching — 完整方案

## 论文价值 — 范式级创新
这改变了 occluded ReID 的匹配范式:
- 标题候选: "Optimal Transport Matching for Occluded Person Re-Identification"
- 贡献1: 将 OT 引入 occluded ReID 匹配（首次）
- 贡献2: Pose-guided transport mass（姿态信息指导传输权重）
- 贡献3: 端到端可微分 Sinkhorn training
