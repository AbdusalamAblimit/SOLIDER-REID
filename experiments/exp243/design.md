# exp243: LGPA — Language-Grounded Part Assignment (CLIP + Pose)

## 核心创新 (范式级)

**首次结合 VLM 语义知识 + 几何 pose 信息 做 occluded person ReID 的 part 表征。**

现有方法:
- KPR: parsing labels 做 part assignment (需要额外的 parsing model)
- ProFD: CLIP text prompts 做 part prototypes (但忽略 pose 几何)
- 我们的 PPA: pose heatmap 做 assignment supervision (但无语义锚定)

**LGPA 的独特之处**: 
1. CLIP frozen text encoder 生成语义 part prototypes (6 个: 5 body parts + background)
2. 这些 prototypes 用 cross-attention 与 backbone spatial tokens 对齐
3. Pose heatmaps 作为 additive bias 注入 attention score (QK^T/sqrt(d) + pose_bias)
4. 三者融合: 语义(CLIP) + 几何(pose) + 视觉(backbone) = 更强的 part 表征

**没有人做过 pose-conditioned CLIP part assignment for ReID。**

## 技术方案

### 架构

```
1. CLIP Text Encoder (frozen):
   texts = ["head and face of a person", "torso and chest of a person", 
            "arms and hands of a person", "upper legs and thighs of a person",
            "lower legs and feet of a person", "background scene"]
   text_protos = CLIP.encode_text(texts)  # (6, 512) → project to (6, 768)

2. Pose-Conditioned Cross-Attention:
   Q = text_protos (6, 768)  — semantic queries
   K = spatial_tokens (48, 768)  — backbone features
   V = spatial_tokens (48, 768)
   
   # Pose bias (additive, injected BEFORE softmax):
   pose_bias = compute_pose_bias(scene_heatmaps)  # (6, 48)
   
   # Attention with pose bias:
   attn = softmax(Q K^T / sqrt(d) + pose_bias)
   part_feats = attn @ V  # (6, 768) — pose-conditioned semantic part features

3. End-to-end training:
   - Part ID loss on each of 5 body part features (exclude background)
   - Pooled part ID loss (visibility-weighted average)
   - Part triplet loss
   - Assignment supervision: KL(pose_GT || pose_conditioned_attn_weights)
   - All gradients flow to backbone (non-detached)

4. Test time:
   - Same cross-attention (CLIP features are fixed)
   - Per-part features for matching (equal_concat: global + pooled + 5 parts)
```

### COCO keypoint → part mapping (5 parts)

| Part | Text prompt | COCO keypoints |
|------|-------------|----------------|
| Head | "head and face of a person" | 0,1,2,3,4 (nose, eyes, ears) |
| Torso | "torso and chest of a person" | 5,6,11,12 (shoulders, hips) |
| Arms | "arms and hands of a person" | 5,6,7,8,9,10 (shoulders, elbows, wrists) |
| Upper legs | "upper legs and thighs of a person" | 11,12,13,14 (hips, knees) |
| Lower legs | "lower legs and feet of a person" | 15,16 (ankles) |
| Background | "background scene" | 1 - max(body_parts) |

### 关键设计

1. **CLIP frozen**: 不训练 CLIP，只用它的文本特征作为语义锚
2. **Text-to-Visual projection**: Linear(512→768) 将 CLIP 特征对齐到 backbone 空间
3. **Pose bias**: 在 attention score 中直接注入 (QK^T/sqrt(d) + pose_bias)
4. **端到端**: cross-attention Q/K/V 投影的梯度流到 backbone
5. **scene_heatmaps**: 与 PPA 基线保持一致 (单变量对照)
6. **CLIP hard-fail**: 如果 open_clip 不可用则直接报错 (不静默退化)

### 与 PPA 的关键区别 (单变量)

- PPA: 每个 token 用线性层分配到 part → 无语义锚定
- LGPA: CLIP 文本原型作为 part queries + pose bias in attention → 语义化的 part 表征
- 两者都用 scene_heatmaps, 5 body parts, 同样的 loss 结构

## 实现文件

1. `model/modules/clip_part_head.py` — 新文件 (手动 cross-attention + pose bias)
2. `model/pose_backbone_model.py` — 集成 (LGPA init + train/test forward)
3. `config/defaults.py` — 配置
4. `processor/processor.py` — LGPA assign_loss

## 对照组
- exp241 (PPA+GCN): 63.7/75.3 (+0.5/-0.1)
- exp237 (PPA only): 63.7/75.0 (+0.5/-0.4)
- exp191 (GCN only): 63.2/75.4

## 预期结果
- 成功: +1.0~2.0% mAP (CLIP 语义增强 part discrimination)
- 失败: ~0% (CLIP 特征与 ReID 特征空间不兼容)
