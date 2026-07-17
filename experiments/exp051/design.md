# 实验 exp051: Pose-Aware Metric Learning (PAML)

## 动机
- **连续 3 个训练端辅助 loss 实验失败**（exp047 CSGT, exp048 SGMKC, exp050 PAMC），说明在 PSG+GCN 上"添加新 loss"的方向已穷尽
- **CVK test-time 匹配已显示可复现正信号**（+0.8-0.9% mAP，2 个 checkpoint 验证）
- **发现 train-test metric mismatch**：训练时 GCN branch 的 triplet loss 使用聚合 skeleton feature 的欧氏距离，但测试时 CVK 使用逐关键点 pairwise 距离（confidence 加权）。模型从未被训练去优化逐关键点距离度量
- **文献支撑**：ICLR 2025 camera bias 论文揭示特征维度间的偏差；P3E 概率嵌入论文表明不确定性建模有价值。更根本地，metric learning 原则要求训练目标与测试度量一致

## 创新点 / 核心想法
- **核心假设**：如果将 GCN branch 的 triplet loss 距离计算从"聚合特征距离"改为"逐关键点 confidence 加权 pairwise 距离"（与 CVK 测试时逻辑一致），模型将学到更适合逐关键点匹配的特征，从而同时提升 equal_concat 和 CVK 的性能
- **关键区别**：这不是添加新的辅助 loss（已证伪 3 次），而是修改已有 part triplet loss 的距离度量函数。不添加新模块、新参数、新超参
- **与 exp036 (per-keypoint triplet) 的区别**：exp036 添加了 17 个独立的 triplet loss（每个关键点一个独立 loss），导致过度约束。PAML 使用单一 triplet loss，但距离通过逐关键点计算后聚合——保持了全局的 hard mining 逻辑

## 技术方案

### 修改文件
1. **`loss/make_loss.py`** — 添加 `_compute_paml_triplet()` 函数
2. **`config/defaults.py`** — 添加 `POSE_PAML = False` 配置开关
3. **`configs/occluded_duke/pose_psg_gcn_paml.yml`** — 实验配置文件

### 核心修改：Part triplet loss 的距离计算

**当前流程**：
```
skeleton_feat = confidence_weighted_average(kp_feats)  # (B, 768)
dist_mat = euclidean_dist(skeleton_feat, skeleton_feat)  # (B, B)
→ hard_example_mining(dist_mat, labels)
→ margin_ranking_loss
```

**PAML 流程**：
```
kp_feats: (B, 17, 768)
kp_weights: (B, 17)  # confidence scores from ViTPose

# 1. 逐关键点计算 pairwise 距离
for k in 0..16:
    dist_k = euclidean_dist(kp_feats[:, k, :], kp_feats[:, k, :])  # (B, B)

# 2. Confidence 加权聚合
# 对每对 (i, j)，权重 = min(score_i_k, score_j_k)
min_weights[i, j, k] = min(kp_weights[i, k], kp_weights[j, k])
dist_mat[i, j] = sum_k(dist_k[i,j] * min_weights[i,j,k]) / sum_k(min_weights[i,j,k])

# 3. 标准 hard mining + margin ranking loss（与原 triplet 完全相同）
→ hard_example_mining(dist_mat, labels)
→ margin_ranking_loss
```

### 数据流
```
Input Image → Swin backbone + PSG → Stage 3 features
                                         ↓
                                    bilinear sample at keypoints
                                         ↓
                                    GCN → enhanced kp_feats (B, 17, 768)
                                         ↓
                               ┌─────────┴──────────┐
                               ↓                    ↓
                        weighted avg pool     PAML distance matrix
                               ↓                    ↓
                          ID loss (CE)      Part triplet loss (margin)
                               ↓                    ↓
                         skeleton_feat    kp_feats (for test-time CVK)
```

### 关键超参数
- `POSE_PAML = True` — 启用 PAML 距离计算
- 无新增超参：使用已有的 kp_weights (ViTPose confidence scores)
- 距离度量：欧氏距离（与当前 triplet loss 一致）
- Hard mining 逻辑：完全复用已有的 `hard_example_mining`
- 注意：只修改 Part triplet loss 的距离计算，Global triplet loss 不变

## 预期结果
- **如果假设成立**：
  - equal_concat mAP 提升 0.3-1.0%（因为关键点特征更适合逐关键点匹配，concat 的表达力增强）
  - CVK test-time 性能进一步提升（因为模型现在直接为 CVK 式匹配而优化）
  - R1 可能不变或小幅提升
- **如果失败**：
  - 最可能原因：逐关键点距离在 batch-hard mining 中的行为与聚合距离差异不大（梯度信号相似）
  - 次要原因：低 confidence 关键点引入距离噪声，影响 hard mining 质量
  - 即使中性，也是有价值的消融实验——证明了距离度量方式不是性能瓶颈

## 对照组
- **Baseline 对照**：exp030a (PSG+GCN, equal_concat) 3-seed mean = 60.73% mAP / 72.57% R1
- **消融变量**：仅修改 Part triplet loss 的距离计算方式（聚合 → 逐关键点），其他所有配置完全相同
- **后续验证**：如果 PAML 有效，在训练后的 checkpoint 上重新测试 CVK（预期增益更大）
