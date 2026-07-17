# 实验 exp038: Adaptive Feature Fusion (AFF)

## 动机

### 问题背景
当前最优方案 (exp030a / exp035a) 使用 PSG+GCN 架构，在测试时将 global_feat 和 skeleton_feat 以 `equal_concat` 模式融合：对两个 768-d 向量分别 L2-normalize 后拼接为 1536-d 特征。这是一个**固定等权**融合方案。

### 存在的不足
1. **Occluded-Duke 中图像差异极大**：有些图像人体完全可见（全身、正面），有些严重遮挡（只露上半身、侧面被遮挡）。对于清晰的全身图像，global_feat (GAP) 包含了完整的外观信息，skeleton_feat 的关键点级特征可能引入冗余甚至噪声（如背景特征被采样到关键点位置）。对于严重遮挡的图像，global_feat 受遮挡物污染，skeleton_feat 只从可见关键点采样，更鲁棒。
2. **等权融合无法适应这种差异**：L2-normalize 后 equal concat 隐式地给了两个分支相同的余弦距离权重。这对所有样本一视同仁，无法根据具体图像的遮挡程度调整。
3. **实验证据支持**：
   - exp030a-global (59.33% mean) vs exp030a-eq (60.73% mean)：fusion 比 global-only 好 +1.40%
   - exp030a-p (57.97% mean)：skeleton 单独使用比 global 差 -1.37%
   - 这说明两个分支的互补性是真实的，但 skeleton 独立能力弱于 global。等权融合给了 skeleton 过高的"投票权"，adaptive weighting 应该让 global 主导、skeleton 补充

### 前序实验教训
- exp022-eq (PDS, equal_concat) 56.1%：5:1 维度比的 equal_concat 灾难性地稀释 global
- exp028 (Part LR 3x)：Part 收敛不是瓶颈，而是 fusion 策略
- exp035b (score*visibility weighting)：在 keypoint pooling 权重上的改进反而有害 -0.7%
- exp036 (per-kp triplet)：增加 loss 维度反而 -0.5%
- exp037 (learnable kp attention)：尚在训练中，改进 pooling 权重

这些实验都在尝试改进 skeleton 分支内部的质量，但没有触及 **两个分支之间的融合方式**。AFF 是第一个直接优化 fusion 策略的实验。

## 创新点 / 核心想法

**核心假设**: 学习一个轻量的 per-sample fusion gate，根据当前图像的 global 和 skeleton 特征自适应地决定两个分支的融合权重，可以比固定等权 concat 更好地利用两个分支的互补性。

**具体地**: 训练一个小 MLP 接收两个分支的 BN 后特征，输出一个标量 alpha (0~1) 表示 global 分支的权重。遮挡严重时 alpha 降低（更依赖 skeleton），全身可见时 alpha 升高（更依赖 global）。训练端为融合后的特征配置独立的 ID loss，迫使模型学会根据图像内容做最优融合。

## 技术方案

### 架构设计

```
                    PSG-enhanced Stage 3 features (B, 768, 12, 4)
                                    |
               ┌────────────────────┴────────────────────┐
               ↓                                         ↓
            GAP → BN                          SkeletonGCNHead
               ↓                              (bilinear sample → GCN → weighted avg)
         global_feat (B, 768)                       ↓
         global_bn (B, 768)                 skeleton_feat (B, 768)
               ↓                            skeleton_bn (B, 768)   ← 新增 BN
               |                                     |
               ├──── classifier_global ──→ loss_g    ├──── classifier_skeleton ──→ loss_s
               |                                     |
               └──────────────┬──────────────────────┘
                              ↓
                     Fusion Gate MLP
                  Input: cat(global_bn, skeleton_bn) → (B, 1536)
                  Output: alpha ∈ (0, 1) via sigmoid → (B, 1)
                              ↓
                  fused_feat = alpha * global_feat + (1-alpha) * skeleton_feat
                              ↓
                         BN_fused → classifier_fused → loss_fused
                              ↓ (test)
                   L2-norm(fused_feat) as test feature

Alternative test mode (backward-compatible):
  - 'equal_concat': L2(global) || L2(skeleton)     ← 当前默认
  - 'adaptive_fused': L2(fused_feat)                ← 新增
  - 'adaptive_concat': L2(alpha*global) || L2((1-alpha)*skeleton) ← 新增
```

### 模块设计: AdaptiveFusionGate

```python
class AdaptiveFusionGate(nn.Module):
    """Per-sample adaptive fusion of global and skeleton features."""

    def __init__(self, feat_dim=768, hidden_dim=256, num_classes=702):
        super().__init__()
        # Fusion gate: predicts alpha from both features
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize to output ~0 → sigmoid(0)=0.5 → equal weight start
        nn.init.zeros_(self.gate[2].weight)
        nn.init.zeros_(self.gate[2].bias)

        # BN + Classifier for fused feature
        self.bn_fused = nn.BatchNorm1d(feat_dim)
        self.bn_fused.bias.requires_grad_(False)
        nn.init.constant_(self.bn_fused.weight, 1.0)
        nn.init.constant_(self.bn_fused.bias, 0.0)

        self.classifier_fused = nn.Linear(feat_dim, num_classes, bias=False)

    def forward(self, global_feat, skeleton_feat, global_bn, skeleton_bn):
        """
        Args:
            global_feat: (B, D) pre-BN global feature
            skeleton_feat: (B, D) pre-BN skeleton feature
            global_bn: (B, D) post-BN global feature (for gate input)
            skeleton_bn: (B, D) post-BN skeleton feature (for gate input)

        Returns:
            alpha: (B, 1) fusion weight for global branch
            fused_feat: (B, D) fused pre-BN feature
            fused_bn: (B, D) fused post-BN feature
            fused_cls: (B, num_classes) classifier logits (training only)
        """
        # Gate input: BN-normalized features (more stable)
        gate_input = torch.cat([global_bn, skeleton_bn], dim=1)  # (B, 2D)
        alpha = torch.sigmoid(self.gate(gate_input))  # (B, 1)

        # Weighted fusion of pre-BN features
        fused_feat = alpha * global_feat + (1 - alpha) * skeleton_feat  # (B, D)

        # BN + Classifier for fused feature
        fused_bn = self.bn_fused(fused_feat)
        fused_cls = self.classifier_fused(fused_bn)

        return alpha, fused_feat, fused_bn, fused_cls
```

### 修改文件清单

1. **`model/modules/adaptive_fusion.py`** (新增)
   - AdaptiveFusionGate 模块定义

2. **`model/modules/skeleton_gcn.py`** (修改)
   - `SkeletonGCNHead.forward()`: 额外返回 `skeleton_bn` 特征（BN 后）
   - 当前只返回 `[skeleton_feat]`（pre-BN），需要也返回 BN 后的版本供 gate 使用
   - 或者：gate 直接用 pre-BN 特征也行（更简单，但训练可能不太稳定）

3. **`model/pose_backbone_model.py`** (修改)
   - `__init__`: 当 `POSE_ADAPTIVE_FUSION=True` 时创建 AdaptiveFusionGate
   - `forward()` 训练路径:
     - 从 skeleton_head 获取 skeleton_feat
     - 计算 skeleton_bn = skeleton_head.bn(skeleton_feat)
     - 调用 fusion_gate 得到 fused feature
     - 返回 cls_scores 列表中增加 fused_cls
     - 返回 feats 列表中增加 fused_feat
   - `forward()` 测试路径:
     - 新增 `adaptive_fused` 模式：直接输出 L2-norm(fused_feat)
     - 新增 `adaptive_concat` 模式：用 alpha 加权后再 L2-norm concat
     - 保留 `equal_concat` 作为默认回退

4. **`config/defaults.py`** (修改)
   - `POSE_ADAPTIVE_FUSION = False`
   - `POSE_AFF_HIDDEN = 256` (gate MLP hidden dim)
   - `POSE_AFF_LOSS_WEIGHT = 1.0` (fused loss 权重)

5. **`loss/make_loss.py`** (可能修改)
   - 当 score 列表长度增加时（新增 fused_cls），loss 函数的 list-loss 路径需要正确处理
   - 当前 list-loss 路径: `[global_cls] + gcn_cls_scores`，新增后变为 `[global_cls] + gcn_cls_scores + [fused_cls]`
   - 需要确认 part_id_avg 和 part_tri_avg 的权重分配是否合理

6. **`configs/occluded_duke/exp038_aff.yml`** (新增)
   - 基于 exp035a config，添加 AFF 相关设置

### 数据流详解 (训练)

```
Input → PSG Backbone → featmap (B, 768, 12, 4)
                         |
            ┌────────────┴────────────────────────────┐
            ↓                                          ↓
         GAP → global_feat (B, 768)        skeleton_head(featmap.detach(), pose_dict)
            ↓                                          ↓
         bottleneck (BN) → global_bn         skeleton_head.bn → skeleton_bn
            ↓                                          ↓
    classifier(global_bn) → cls_global      skeleton_head.classifier(skeleton_bn) → cls_skeleton
            ↓                                          ↓
            └────────────┬─────────────────────────────┘
                         ↓
              fusion_gate(global_feat, skeleton_feat, global_bn, skeleton_bn)
                         ↓
              alpha, fused_feat, fused_bn, fused_cls
                         ↓
            Loss = w_g * L(cls_global, global_feat) +
                   w_p * L(cls_skeleton, skeleton_feat) +
                   w_f * L(fused_cls, fused_feat)

其中 w_g, w_p 由现有 POSE_PART_WEIGHT 控制
w_f 由新增 POSE_AFF_LOSS_WEIGHT 控制
```

### 数据流详解 (测试)

```
mode='adaptive_fused':
  alpha, fused_feat, _, _ = fusion_gate(global_feat, skeleton_feat, global_bn, skeleton_bn)
  test_feat = L2_normalize(fused_feat)   # (B, 768)

mode='adaptive_concat':
  alpha, _, _, _ = fusion_gate(global_feat, skeleton_feat, global_bn, skeleton_bn)
  test_feat = cat(L2(alpha * global_feat), L2((1-alpha) * skeleton_feat))   # (B, 1536)

mode='equal_concat' (default, backward-compatible):
  test_feat = cat(L2(global_feat), L2(skeleton_feat))   # (B, 1536)
```

### 关键设计决策

1. **Gate 输入用 BN 后特征**: BN 后特征归一化分布更稳定，有利于 gate 学习。但 fusion 操作在 pre-BN 空间进行（因为 fused_feat 需要过自己的 BN）。

2. **Zero-init gate**: 让初始 alpha=0.5（sigmoid(0)），等价于 equal weight 融合。训练开始时不破坏两个分支的学习，gate 逐渐学习最优权重。

3. **Skeleton 分支仍然 detach**: `featmap.detach()` 传入 skeleton_head，避免 skeleton 梯度回传到共享 backbone。这一点与 exp035a 保持一致。

4. **Fused loss 独立于 global/skeleton loss**: 不替换现有的 global 和 skeleton loss，而是新增一个 fused loss。这确保了两个分支仍然被独立优化（特别是 skeleton 分支仍需 detach 后的特征来训练），fused loss 只优化 gate 和 fused BN/classifier。

5. **Gate 的梯度流**:
   - gate 参数通过 fused_loss 的反向传播更新
   - gate 的输入是 global_bn 和 skeleton_bn（都是计算图中的节点），所以 fused_loss 的梯度**会**回传到 global BN 和 skeleton BN
   - **但 skeleton_feat 是从 detached featmap 计算的**，所以梯度不会进一步回传到 backbone
   - global_feat 的梯度会正常回传到 backbone（这是期望的行为）

6. **adaptive_fused 测试模式输出 768-d 而非 1536-d**:
   - 优势：更紧凑的特征，检索更快
   - 风险：可能丢失 equal_concat 的互补信息
   - 因此保留 adaptive_concat 作为 1536-d 的替代方案

### 关键超参数

| 参数 | 值 | 选择依据 |
|------|-----|---------|
| POSE_ADAPTIVE_FUSION | True | 开关 |
| POSE_AFF_HIDDEN | 256 | 128 太小可能表达力不足，256 与 GCN hidden 一致 |
| POSE_AFF_LOSS_WEIGHT | 1.0 | 初始与 global/skeleton loss 等权，后续可调 |
| gate init | zeros | sigmoid(0)=0.5 → equal weight 初始化 |
| gate 参数量 | 768*2*256 + 256 + 256*1 + 1 = ~394K | 相对整体模型的 ~500K 额外参数（GCN部分），约 80% 增加，仍可接受 |

## 预期结果

### 乐观 (+0.5~1.0% mAP)
- adaptive_fused: 61.5~62.0% mAP（768-d，检索更快且性能更好）
- adaptive_concat: 61.5~62.0% mAP（1536-d）
- 原因：gate 学会了在遮挡图像上依赖 skeleton，在清晰图像上依赖 global，比固定权重更优

### 现实 (+0.1~0.5% mAP)
- adaptive_fused: 61.0~61.5% mAP
- adaptive_concat: 61.2~61.6% mAP
- 原因：fusion gate 有轻微改善但受限于训练数据量和 gate capacity

### 悲观 (0 or -0.3% mAP)
- 原因：gate 退化为 alpha=0.5 常数（等价于 equal weight），额外参数增加过拟合
- 或者：fused_loss 的梯度干扰了 global/skeleton 分支的训练

### 失败风险分析
1. **Gate 退化**: 如果 gate 输出始终 ~0.5，说明两个特征对 gate 没有提供足够的区分信号。缓解: 可以增加 pose 信息（如 avg keypoint score）作为 gate 额外输入
2. **梯度冲突**: fused_loss 的梯度同时影响 global BN 和 skeleton BN。如果这导致主分支性能下降，需要 detach gate 输入。缓解: 在 gate 输入上 `.detach()`
3. **过拟合**: 394K 额外参数可能导致小数据集上过拟合。缓解: 降低 AFF_HIDDEN (如 128 或 64)

## 对照组

- **Baseline**: exp035a (PSG + GCN, score weight, equal_concat) = 61.1% mAP / 73.8% R1
- **3-seed 参考**: exp030a equal_concat 3-seed mean = 60.73% mAP / 72.57% R1
- **消融变量**: 仅在 exp035a 基础上增加 AdaptiveFusionGate 模块 + fused loss

### 消融路径
如果 AFF 有效，后续消融:
- AFF w/o fused loss（gate 只用于测试，训练时不加 fused_cls loss）
- AFF w/ detached gate input（gate 输入不回传梯度）
- AFF hidden dim 消融 (64 / 128 / 256 / 512)
- AFF + pose info（gate 输入增加 avg keypoint score）

### 与论文 story 的关系
如果成功，AFF 可以作为论文贡献之一：
- PSG: backbone 级 pose 注入 → 改善特征质量
- Skeleton GCN: 骨架拓扑特征传播 → 提供互补的结构化特征
- **AFF: 自适应融合 → 根据图像遮挡程度动态调整分支权重**
- 三者构成完整的"从特征提取到融合"的 pose-guided pipeline

如果失败（alpha 退化为常数），则反过来证明"equal weight concat 已经接近最优"，这本身也是有价值的负面结果，可以支撑 equal_concat 作为简洁且鲁棒的 fusion 选择。
