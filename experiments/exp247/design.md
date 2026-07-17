# exp247: VCSR — Visibility-Conditional Semantic Routing

## 核心创新 (范式级)

**重新定义问题**: "Occluded ReID fails because fixed part vocabularies assume complete semantic support. Under occlusion, the model should instantiate only the semantic groups actually supported by visible evidence."

现有方法 (ProFD, KPR, PPA, LGPA) 都用固定数量的 parts:
- 遮挡图像: 5 parts → 2-3 个 part features 是噪声 (遮挡区域)
- 完整图像: 5 parts → 全部有效
- 匹配: fixed-length concat → 噪声 parts 污染距离

**VCSR 的革命**: 不输出固定数量的 part features, 而是只输出可见的 parts:
- 遮挡图像: 3 visible parts → 3 features (无噪声)
- 完整图像: 5 visible parts → 5 features (全有效)
- 匹配: 只比较共同可见的 parts (set-to-set, 类似 MaxSim)

**没有人在训练端做过 visibility-conditional part routing + set matching**。

## 技术方案

### 架构

```
1. CLIP Part Prototypes (frozen, 5 body parts + background)
   → 与 LGPA 相同

2. Visibility Gating (新!):
   vis_score[k] = pose_heatmaps → per-part visibility score (0-1)
   active_mask = vis_score > threshold (binary, per-sample per-part)
   
   训练时: 只对 active parts 计算 ID loss 和 triplet loss
   被遮挡的 parts: 不参与任何 loss (不是 masking 到 0, 是完全不算)

3. Pose-Conditioned Cross-Attention (与 LGPA 相同):
   → 只为 active parts 计算 cross-attention
   → attn = softmax(QK^T/sqrt(d) + pose_bias)

4. Set-to-Set Matching (训练端创新):
   - 训练 triplet loss: MaxSim distance on active parts only
   - 训练 ID loss: 只对 active parts 算 per-part CE
   - 测试: variable-length features + MaxSim matching
   
   这统一了 MaxSim (之前只是 test-time trick) 为 end-to-end training objective!
```

### 关键创新点 (与 prior art 的区分)

1. **vs ProFD/LGPA**: 固定 parts → 动态 parts, 无噪声 part features
2. **vs KPR**: KPR 用 visibility 做 attention weighting, 仍输出固定长度. VCSR 完全跳过遮挡 parts.
3. **vs MaxSim**: MaxSim 只在 test 时用. VCSR 在训练时就用 MaxSim distance, end-to-end.
4. **vs Visibility weighting**: weighting 仍保留遮挡 features. VCSR 完全移除, 从根本上消除噪声.

### 实现细节

1. `model/modules/vcsr_head.py` — 新文件:
   - 继承 CLIPPartHead 的 CLIP prototypes + cross-attention
   - 添加 VisibilityGate: pose heatmaps → per-part binary mask
   - 训练: 只对 active parts 输出 cls_score 和 feat
   - MaxSim triplet loss 集成

2. `model/pose_backbone_model.py` — VCSR 路径
3. `processor/processor.py` — visibility-aware loss
4. `loss/make_loss.py` — MaxSim-aware triplet

## 对照组
- exp244 (LGPA-D, fixed 5 parts): 65.3/75.7
- exp191 (GCN, fixed pooled): 63.2/75.4

## 预期结果
- 成功: +1~3% mAP over LGPA-D (VCSR 消除遮挡噪声, 应提升匹配质量)
- 消融: VCSR vs LGPA-D 证明动态 routing 的价值
