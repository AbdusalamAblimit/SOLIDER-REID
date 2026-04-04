# exp243: LGPA — Language-Grounded Part Assignment (CLIP + Pose)

## 核心创新 (范式级)

**首次结合 VLM 语义知识 + 几何 pose 信息 做 occluded person ReID 的 part 表征。**

现有方法:
- KPR: parsing labels 做 part assignment (需要额外的 parsing model)
- ProFD: CLIP text prompts 做 part prototypes (但忽略 pose 几何)
- 我们的 PPA: pose heatmap 做 assignment supervision (但无语义锚定)

**LGPA 的独特之处**: 
1. CLIP frozen text encoder 生成语义 part prototypes ("head", "torso", "arms", "legs")
2. 这些 prototypes 用 cross-attention 与 backbone spatial tokens 对齐
3. Pose heatmaps 提供空间约束 (哪些位置属于哪个 part)
4. 三者融合: 语义(CLIP) + 几何(pose) + 视觉(backbone) = 更强的 part 表征

**没有人做过 pose-conditioned CLIP part assignment for ReID。**

## 技术方案

### 架构

```
1. CLIP Text Encoder (frozen):
   texts = ["head of a person", "torso of a person", "arms of a person", 
            "legs of a person", "background"]
   text_protos = CLIP.encode_text(texts)  # (5, 512) → project to (5, 768)

2. Cross-Attention Part Assignment:
   Q = text_protos (5, 768)  — semantic queries
   K = spatial_tokens (48, 768)  — backbone features
   V = spatial_tokens (48, 768)
   
   # Pose-conditioned attention mask:
   # For each part query, mask tokens that don't correspond to that body part
   pose_mask = _compute_pose_mask(heatmaps)  # (5, 48) binary
   
   # Cross-attention with pose mask:
   attn = softmax(Q K^T / sqrt(d) + pose_mask_bias)
   part_feats = attn @ V  # (5, 768) — semantic-grounded part features

3. End-to-end training:
   - Part ID loss on each of 5 part features
   - Part triplet loss
   - Assignment supervision from pose heatmaps
   - All gradients flow to backbone

4. Test time:
   - Same cross-attention (CLIP features are fixed)
   - Per-part features for matching (MaxSim or equal_concat)
```

### 关键设计

1. **CLIP frozen**: 不训练 CLIP，只用它的文本特征作为语义锚
2. **Text-to-Visual projection**: Linear(512→768) 将 CLIP 特征对齐到 backbone 空间
3. **Pose mask**: 限制 cross-attention — "head" query 只看 head 区域 tokens
4. **端到端**: cross-attention 的梯度流到 backbone

### 与 PPA 的关键区别

- PPA: 每个 token 用线性层分配到 part → 无语义锚定
- LGPA: CLIP 文本原型作为 part queries → 语义化的 part 表征
- LGPA 的 part features 有 CLIP 的语义先验，更 robust

## 实现文件

1. `model/modules/clip_part_head.py` — 新文件
2. `model/pose_backbone_model.py` — 集成
3. `config/defaults.py` — 配置
4. `processor/processor.py` — loss

## 对照组
- exp241 (PPA+GCN): 63.7/75.3 (+0.5/-0.1)
- exp191 (GCN only): 63.2/75.4

## 预期结果
- 成功: +1.0~2.0% mAP (CLIP 语义增强 part discrimination)
- 失败: ~0% (CLIP 特征与 ReID 特征空间不兼容)
