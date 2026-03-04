# Paper 1: KPR -- Keypoint Promptable Re-Identification
**来源**: ECCV 2024
**仓库**: https://github.com/VlSomers/keypoint_promptable_reidentification
**arXiv 摘要**: KPR is a Swin Transformer-based part-based ReID model that uses keypoint prompts (positive and negative) to resolve multi-person ambiguity in occluded scenarios. It produces K part-based embeddings with visibility scores, enabling robust matching by only comparing mutually visible parts. Built on top of BPBReID (WACV23), it achieves SOTA on Occluded-Duke, Occluded-ReID, Market-1501, and the new Occluded-PoseTrack-ReID dataset.

## 代码架构概览

### 核心文件
- **模型主体**: `torchreid/models/kpr.py` -- KPR model class, part-based pooling, classifiers, dim reduction
- **Prompt Tokenizer (核心创新)**: `torchreid/models/promptable_transformer_backbone.py` -- `_mask_embed()` method, defines how keypoint heatmaps are fused with image tokens
- **SOLIDER Swin backbone wrapper**: `torchreid/models/promptable_solider.py` -- `PromptableSoliderSwinTransformer`, wraps SwinTransformer with prompt injection
- **Swin Transformer**: `torchreid/models/solider/backbones/swin_transformer.py` -- PatchEmbed, SwinTransformerBlock, PatchMerging
- **训练引擎**: `torchreid/engine/image/part_based_engine.py` -- forward_backward, loss combination, feature extraction
- **GiLt Loss**: `torchreid/losses/GiLt_loss.py` -- Global-identity Local-triplet loss strategy
- **Part Triplet Loss**: `torchreid/losses/part_averaged_triplet_loss.py` -- Part-averaged batch-hard triplet
- **Body Part Attention Loss**: `torchreid/losses/body_part_attention_loss.py` -- Pixel-level part classification loss
- **Keypoints to Masks**: `torchreid/data/datasets/keypoints_to_masks.py` -- Convert keypoint (x,y,c) to gaussian heatmaps
- **Mask Transforms**: `torchreid/data/masks_transforms/` -- Grouping strategies (cck8, cck6, etc.)
- **Distance Computation**: `torchreid/metrics/distance.py` -- Visibility-aware part-based distance

### 模型入口
`torchreid/models/kpr.py::KPR.__init__()` and `forward()`. The backbone is instantiated via `models.build_model(self.model_cfg.backbone, ...)`, which for SOLIDER creates a `PromptableSoliderSwinTransformer`.

### 数据流 (Forward Pass)
```
Input: images [N,3,384,128], prompt_masks [N,K+2,384,128], target_masks [N,K+1,Hf,Wf]
  |
  v
PromptableSoliderSwinTransformer.forward():
  1. Swin PatchEmbed(images) -> image_tokens [N, H/4*W/4, 128]
  2. _mask_embed():  masks_patch_embed(prompt_masks) -> prompt_tokens [N, H/4*W/4, 128]
     image_tokens += prompt_tokens  (additive fusion at token level)
  3. _cam_embed(): + SIE camera embedding
  4. Run through 4 Swin stages -> spatial_features [N, 1024, H/32, W/32]
  |
  v
KPR.forward():
  5. Optional dim_reduce (1x1 Conv: 1024->512) -> [N, 512, 12, 4]
  6. PixelToPartClassifier -> pixels_cls_scores [N, K+1, 12, 4]
     softmax -> pixels_parts_probabilities (learned attention maps)
  7. Mask-weighted pooling:
     - GlobalAvgPool(features) -> global_embedding [N, D]
     - GWAP(features, fg_mask) -> foreground_embedding [N, D]
     - GWAP(features, bg_mask) -> background_embedding [N, D]
     - GWAP(features, parts_masks) -> parts_embeddings [N, K, D]
  8. BNClassifier for each embedding type -> cls_scores + bn_embeddings
  9. Visibility scores from attention map activations (argmax -> one_hot -> amax)
  |
  v
Output: embeddings_dict, visibility_scores_dict, id_cls_scores_dict, pixels_cls_scores
```

## 可拆解模块清单

### 模块 A: Prompt Tokenizer (Keypoint Prompt Fusion)
- 文件位置: `torchreid/models/promptable_transformer_backbone.py` L110-L148
- 功能: 将关键点 heatmap 转换为 token 并与图像 token 相加，实现 keypoint prompt 的注入
- **核心机制**:
  - `embed_heatmaps_patches` 策略 (主要策略):
    1. 输入 prompt_masks [N, K+2, H, W] (K 个 body part heatmap + 1 background + 1 negative keypoints)
    2. 通过独立的 `masks_patch_embed` (与图像 PatchEmbed 结构相同的 Conv2d, kernel=4, stride=4) 将 heatmap 转为 tokens
    3. 将 prompt tokens 加到 image tokens 上: `image_features += part_tokens`
  - `spatialize_part_tokens` 策略 (备选):
    1. 为每个 part 学习一个可学习的 embedding (`parts_embed` [K+2, 1, embed_dim])
    2. 对 prompt_masks 做 argmax 得到每个 patch 属于哪个 part
    3. 将对应的 part embedding 加到对应位置的 image token 上
- 输入: image_features [N, H/4*W/4, 128], prompt_masks [N, K+2, H, W]
- 输出: image_features [N, H/4*W/4, 128] (same shape, values modified by addition)
- 依赖: 离线提取的 PifPaf keypoints (JSON), 在线转为 gaussian heatmaps
- **关键初始化**: `masks_patch_embed` 权重初始化为零 (`mask_path_emb_init_zeros=True`)，这意味着训练开始时 prompt 不起作用，模型先学会无 prompt 的 ReID，然后逐渐学习利用 prompt
- **移植到我们框架的可行性**: 高
- **额外显存开销估算**: ~50MB (一个额外的 PatchEmbed: Conv2d(K+2, 128, 4, 4) + LayerNorm)。非常轻量。
- **移植方案**:
  1. 我们的 Swin-Tiny embed_dim=96 (不是128)，需要调整 masks_patch_embed 的 out_channels=96
  2. 在 `swin_transformer.py` 的 forward 中，patch_embed 之后、进入 stages 之前，加入 prompt token 的加法
  3. prompt_masks 的通道数 = prompt_parts_num + 1(background) + 1(negative) = K+2
  4. 可以先用 K=8 (cck8 grouping: head, left_arm, right_arm, torso, left_leg, right_leg, left_feet, right_feet) + bg + neg = 10 channels

### 模块 B: Learnable Part Attention Head (PixelToPartClassifier)
- 文件位置: `torchreid/models/kpr.py` L459-L479
- 功能: 1x1 Conv 将 spatial features 分类为 K+1 个 body part (K parts + background)，softmax 生成 attention maps
- 输入: spatial_features [N, D, Hf, Wf]
- 输出: pixels_cls_scores [N, K+1, Hf, Wf], after softmax -> part attention maps
- 依赖: 无
- **核心设计**:
  - BN -> Conv2d(D, K+1, 1x1) -> softmax
  - 训练时用 target_masks (来自 PifPaf 人体解析) 作为 GT 监督 pixels_cls_scores
  - 推理时不需要 external masks，模型自己预测 attention maps
  - 权重初始化: Conv2d weights init Normal(0, 0.001)
- **移植到我们框架的可行性**: 高
- **额外显存开销估算**: ~10KB (Conv2d(768, K+1, 1))。几乎可以忽略。
- **移植方案**: 直接在我们的模型中 backbone 输出之后加一个 PixelToPartClassifier。我们已有 PAMS 模块做类似的事，可以整合或替换。

### 模块 C: Global Weighted Average Pooling (GWAP)
- 文件位置: `torchreid/models/kpr.py` L572-L586
- 功能: 用 attention masks 对 spatial features 做加权平均池化，提取 part-specific embeddings
- 输入: features [N, D, Hf, Wf], part_masks [N, K, Hf, Wf]
- 输出: parts_features [N, K, D]
- **核心计算**:
  ```python
  parts_features = torch.mul(part_masks.unsqueeze(2), features.unsqueeze(1))  # [N, K, D, Hf, Wf]
  parts_features = torch.sum(parts_features, dim=(-2, -1))  # [N, K, D]
  parts_features_avg = parts_features / part_masks_sum.clamp(min=1e-6)  # [N, K, D]
  ```
  即: 对每个 part，用该 part 的 attention map 对 feature map 做加权求和再归一化
- **移植到我们框架的可行性**: 高
- **额外显存开销估算**: 0 (纯计算，无参数)。但中间张量 [N, K, D, Hf, Wf] 在 K 和 D 较大时占显存。K=8, D=768, Hf=12, Wf=4 -> 每 sample ~1.2MB, batch=32 -> ~38MB。可接受。
- **移植方案**: 我们目前的 PAMS 也有类似的 part pooling。可以参考此实现改进，特别是加权求和归一化的方式比简单 masked GAP 更好。

### 模块 D: GiLt Loss (Global-identity Local-triplet)
- 文件位置: `torchreid/losses/GiLt_loss.py` L11-L121
- 功能: 组合多种 loss 用于不同类型的 embeddings
- **核心策略**:
  ```
  Global embedding:     ID loss (weight=1.0),  Triplet loss (weight=0.0)
  Foreground embedding: ID loss (weight=1.0),  Triplet loss (weight=0.0)
  Concat parts:         ID loss (weight=1.0),  Triplet loss (weight=0.0)
  Parts embeddings:     ID loss (weight=0.0),  Triplet loss (weight=1.0)
  Pixels:               CE loss (weight=0.35)
  ```
  即: 全局特征用 ID 分类 loss, 局部 part 特征用 triplet loss, 像素级用 part attention 分类 loss
- **理由**: Part embeddings 不适合用 ID loss (因为单个 part 信息不足以区分 ID)，但适合用 triplet 拉近同 ID 的 part、推远不同 ID 的 part
- **移植到我们框架的可行性**: 高
- **额外显存开销估算**: 0 (纯 loss 计算)
- **移植方案**: 我们已有 ID loss + triplet loss。需要增加:
  1. 对 concat_parts 和 foreground 单独加 ID loss
  2. 对 parts embeddings 加 part-averaged triplet loss (而非 per-part ID loss)
  3. 加 Body Part Attention Loss (pixel classification CE)

### 模块 E: Part-Averaged Triplet Loss
- 文件位置: `torchreid/losses/part_averaged_triplet_loss.py` L10-L225
- 功能: 对 K 个 part embeddings 的距离矩阵先求 mean，再做标准 batch-hard triplet loss
- **核心流程**:
  1. 计算每个 part 的 pairwise distance [K, N, N]
  2. 用 visibility masks 过滤 (无共同可见 part 的 pair 距离设为 -1)
  3. 对 K 个 distance matrix 做 masked_mean -> combined pairwise distance [N, N]
  4. 在 combined distance 上做 batch-hard mining (hardest positive, hardest negative)
  5. 支持 hard margin 和 soft margin triplet loss
- **与标准 triplet 的区别**: 先平均 K 个 part 距离，再 mine hardest triplet (而非对每个 part 分别 mine)
- **移植到我们框架的可行性**: 高
- **额外显存开销估算**: 0 (纯计算)
- **移植方案**: 可以直接作为独立模块引入。我们当前的 triplet 是在全局特征上做的，这里需要在 [N, K, D] 上做。

### 模块 F: Body Part Attention Loss (Pixel-level Part Segmentation Loss)
- 文件位置: `torchreid/losses/body_part_attention_loss.py` L12-L61
- 功能: 监督 PixelToPartClassifier 的输出，使 attention maps 对齐外部 human parsing labels
- **核心机制**:
  - target: target_masks.argmax(dim=1) -> [N, Hf, Wf] 每个像素的 GT part index
  - prediction: pixels_cls_scores [N, K+1, Hf, Wf]
  - loss: CrossEntropyLoss with label smoothing (default)
  - `best_pred_ratio=1.0`: 只取 loss 最小的 100% 像素参与 loss (可设为 <1.0 过滤噪声标签)
  - weight=0.35 in overall loss
- **移植到我们框架的可行性**: 高
- **额外显存开销估算**: 0
- **移植方案**: 需要有离线提取的 human parsing masks 作为 GT。可以用 PifPaf keypoints 生成 body part masks (KPR 就是这么做的)。

### 模块 G: Visibility-Aware Distance (Test-time)
- 文件位置: `torchreid/metrics/distance.py` L87-L220
- 功能: 在 query-gallery 距离计算时考虑 part visibility scores
- **三种模式**:
  1. 无 visibility scores: 所有 K 个 part distance 直接 mean
  2. Boolean visibility (binary): 只计算双方都可见的 parts 的距离平均值，不可见的标记为 -1 排除
  3. Continuous visibility: 用 sqrt(vis_q * vis_g) 对距离加权平均
- **关键**: `masked_mean` 函数，计算 `sum(dist * mask) / sum(mask)`，当 mask 全 0 时返回 -1
- **移植到我们框架的可行性**: 高 (我们已有 visibility-aware distance 的初步实现)
- **额外显存开销估算**: 0
- **移植方案**: 我们已有类似机制。可以参照 KPR 的实现改进，特别是 binary vs. continuous visibility 的选择。

### 模块 H: Batch-wise Inter-Person Occlusion (BIPO) Data Augmentation
- 文件位置: `torchreid/data/data_augmentation/batch_wise_inter_person_occlusion.py` L20-L99
- 功能: 训练时人为制造多人遮挡，从 batch 中采样其他人的图像覆盖到当前图像上
- **核心流程**:
  1. 从同 batch 其他 ID 的图像中随机选一个 occluder
  2. 用 occluder 的 segmentation mask 将 occluder 的人体区域叠加到目标图像上
  3. 更新目标图像的 human parsing labels (被遮挡区域清零)
  4. 将 occluder 的所有可见 keypoints 作为 negative prompts 添加
- **移植到我们框架的可行性**: 中 (需要改 dataloader 和 augmentation pipeline)
- **额外显存开销估算**: 0
- **移植方案**: 需要比较大的 dataloader 改动。可以先不做，后期如果需要提升 occlusion robustness 再考虑。

### 模块 I: Multi-Stage Fusion (MSF)
- 文件位置: `torchreid/models/kpr.py` L338-L368
- 功能: 将 Swin 各 stage 的特征图 resize 到相同大小后 concat，再 1x1 Conv 降维
- **核心**: 用高分辨率特征 (stage 1-3) 补充低分辨率但语义丰富的 stage 4 特征
- 输入: dict of features_per_stage: {0: [N,C0,H0,W0], 1: [N,C1,H1,W1], ...}
- 输出: fused_features [N, C_out, H0, W0]
- **移植到我们框架的可行性**: 中
- **额外显存开销估算**: ~200MB (所有 stage features 需要 resize + concat，中间张量较大)
- **移植方案**: 需要修改 backbone 返回多 stage 输出。Swin-Tiny 的 4 stages dims 为 [96, 192, 384, 768]，concat 后 1440 维再降到 768。但显存开销较大，可能不适合 with_cp 场景。

### 模块 J: Prompt-Optional Design (Zero-init)
- 文件位置: `torchreid/models/promptable_solider.py` L65-L66, `torchreid/models/promptable_transformer_backbone.py` L119-L124
- 功能: masks_patch_embed 权重初始化为零 + 训练时随机 drop keypoints
- **核心思想**:
  1. 零初始化: 训练开始时 prompt tokens 全为零，模型行为等价于无 prompt 的 baseline
  2. 训练时 drop: `DropRandomKeypoints(p=0.2, ratio=0.5)` 和 `DropAllKeypoints(p=0.3)` 使模型学会在有/无 prompt 时都能工作
  3. 推理时可选: 有 prompt 时用来解决多人歧义，无 prompt 时退化为标准 part-based ReID
- **移植到我们框架的可行性**: 高
- **移植方案**: 简单地对 masks_patch_embed 做 zero_init，并在数据增强时加入 keypoint dropping

## 损失函数

### 1. GiLt Loss (Global-identity Local-triplet)
- **组合策略**:
  - `Global`: ID loss x1.0 (CE with label smoothing, eps=0.1)
  - `Foreground`: ID loss x1.0
  - `Concat Parts`: ID loss x1.0
  - `Parts`: Triplet loss x1.0 (part-averaged triplet, margin=0.3)
  - `Pixels`: CE loss x0.35 (body part classification)
- **思想**: 全局/前景/concat 特征用 ID loss 学习判别性，part 特征用 triplet loss 学习细粒度对齐
- 可否直接用: 可以。我们已有 ID loss 和 triplet loss，需要增加 loss 的应用目标和权重分配。

### 2. Part-Averaged Triplet Loss
- **公式**:
  ```
  d(a,b) = mean_k(||e_a^k - e_b^k||) (only over mutually visible parts)
  L = mean_i[max(0, d(a_i, p_i) - d(a_i, n_i) + margin)]
  ```
  where (a_i, p_i, n_i) are batch-hard mined triplets based on combined part distances.
- margin = 0.3 (default), also supports soft margin variant
- 可否直接用: 可以，但我们的 triplet 需要从全局改为 part-based。

### 3. Body Part Attention Loss (Pixel Classification CE)
- **公式**: 标准 CE with label smoothing + optional top-k filtering
- **监督**: 每个 spatial 位置 (pixel) 预测属于 K+1 个 part 的概率，GT 来自外部 human parsing masks
- weight = 0.35 in total loss
- 可否直接用: 需要有 human parsing GT。KPR 用 PifPaf 离线提取，我们也可以用类似方案。

### 4. Cross Entropy Loss with Label Smoothing
- eps = 0.1
- `targets = (1 - eps) * one_hot + eps / K`
- 可否直接用: 我们已有。

## 训练 Tricks

### 1. 超参数配置 (Occluded-Duke)
- **Backbone**: Swin-Base (SOLIDER pretrained), depths=(2,2,18,2), embed_dim=128
- **Input**: 384 x 128
- **Optimizer**: SGD + cosine annealing warmup (lr=0.008 base, 0.0002 reduced for backbone)
- **Scheduler**: cosine annealing with 5 epochs warmup
- **Epochs**: 120
- **Batch size**: 64
- **Sampler**: RandomIdentitySampler, 4 instances per ID
- **Mixed precision**: True (AMP)
- **Weight decay**: 1e-4

### 2. 关键 Training Tricks
- **Fixbase**: 前 10 个 epoch 冻结 backbone，只训练新加的 heads/prompt tokenizer
- **SOLIDER semantic_weight**: 0.2 (平衡 semantic 和 appearance features)
- **Norm**: mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5] (SOLIDER 专用)
- **Camera embedding**: SIE with cam_num=8, sie_coe=3
- **Dim reduce**: after_pooling, 1024->512 (Swin-Base 最后 stage dim=1024)
- **Pooling**: GWAP (Global Weighted Average Pooling)

### 3. Prompt 相关
- **Keypoint source**: PifPaf (17 keypoints + confidence)
- **Prompt heatmap**: keypoints_gaussian (高斯核, scale=11)
- **Prompt grouping**: cck8 (8 body parts from COCO keypoints)
- **Background token**: 保留 (no_background_token=False)
- **Negative keypoints**: 启用 (use_negative_keypoints=True)
- **Prompt tokenizer**: embed_heatmaps_patches (Conv2d patchification)
- **Prompt tokenizer init**: 零初始化 (mask_path_emb_init_zeros=True)

### 4. 数据增强
- Random flip (rf)
- Random crop (rc)
- Random erasing (re)
- BIPO (Batch-wise Inter-Person Occlusion, p=0.2)
- DropRandomKeypoints (p=0.2, ratio=0.5)
- DropAllKeypoints (p=0.3)
- Color jitter (brightness=0.2, contrast=0.15)

### 5. 测试配置
- **Test embeddings**: ['bn_foreg', 'parts'] -- 前景 BN 特征 + K 个 part 特征
- **Visibility**: binary at both train and test (training_binary_visibility_score=True, testing_binary_visibility_score=True)
- **Mask filtering**: off at training, on at testing
- **Feature normalization**: L2 normalize before computing euclidean distance
- **Distance combine**: mean of visible parts

### 6. MSF (Multi-Stage Fusion)
- `enable_msf=True`: 开启多阶段特征融合
- 将 Swin 4 个 stage 的输出 resize 到相同大小 concat，再 1x1 Conv 降维

## 对我们框架的改进建议

### 优先级最高 -- 直接可移植的模块

1. **引入 Learnable Part Attention (PixelToPartClassifier) + Body Part Attention Loss**
   - 我们已有 PAMS 做 part-based features，但 KPR 的 attention head + CE supervision 方式更成熟
   - 实现: Conv2d(768, K+1, 1) + softmax + CE loss (weight=0.35)
   - 需要: PifPaf/DWPose 离线生成 target masks
   - 预期: 让 part attention maps 更精准地对齐到人体部件，提高 occluded 场景性能

2. **引入 GiLt Loss 策略**
   - 修改我们的 loss：global/foreground/concat 只用 ID loss, parts 只用 triplet loss
   - 用 part-averaged triplet (先 mean K 个 part distance 再 batch-hard mine) 替换当前的 triplet
   - 预期: 更合理的 loss 分配，避免对 part embedding 强制 ID classification

3. **引入 Visibility-Aware Distance (test-time)**
   - 我们已有初步实现，参照 KPR 的 binary/continuous 两种模式完善
   - Binary mode: 只对双方都可见的 parts 计算距离平均值
   - 预期: 对 occluded query 的 mAP 提升显著

### 优先级中等 -- 需要更多工程但回报高

4. **引入 Prompt Tokenizer (Keypoint Prompt)**
   - 对 Swin-Tiny (embed_dim=96): 新建 masks_patch_embed Conv2d(K+2, 96, 4, 4)
   - 零初始化，与图像 tokens 相加
   - 需要: PifPaf 离线提取 keypoints, 在线转 gaussian heatmaps
   - 预期: 解决多人遮挡的歧义问题 (Occluded-Duke 上很关键)

5. **GWAP Pooling**
   - 用加权平均池化替代简单 masked GAP: `sum(mask * feature) / sum(mask)`
   - 对小区域 parts (如手/脚) 更鲁棒

### 优先级较低 -- 工程量大或显存风险

6. **Multi-Stage Fusion (MSF)**
   - 融合 Swin 所有 stage 的特征，提高 part attention 的空间分辨率
   - 但 Swin-Tiny 4 stages concat 后 96+192+384+768=1440 维，需要额外 1x1 Conv 降维
   - 显存开销: 所有中间 feature maps 要保存，with_cp 场景下可能紧张

7. **BIPO Data Augmentation**
   - 需要大幅修改 dataloader，工程量大
   - 如果 keypoint prompt 效果好，可以考虑后期加入

### 与我们 PAMS 模块的对比

| 方面 | 我们的 PAMS | KPR |
|------|------------|-----|
| Part 定义 | 固定 6 parts (horizontal + predefined) | 可学习的 K+1 parts attention (softmax) |
| Attention 来源 | Feature projection + split | PixelToPartClassifier (1x1 Conv, supervised by GT masks) |
| Pooling | Masked average | GWAP (weighted average, mask-area normalized) |
| Loss on parts | Per-part ID loss | Part-averaged triplet loss (更适合 parts) |
| Visibility | 来自 attention activation | 来自 attention argmax (binary) 或 max (continuous) |
| 人体解析 GT | 不需要 | 需要离线 PifPaf masks |
| Keypoint prompt | 无 | 有 (prompt tokenizer, additive fusion) |
| Test-time | 无 prompt | 可选 prompt (zero-init 使两种模式兼容) |

### 具体实施路线建议

**Phase 2a (exp002-003)**: 引入 KPR 的 Learnable Part Attention + GiLt loss 策略
- 保留 Swin-Tiny backbone 不变
- 加 PixelToPartClassifier + Body Part Attention Loss
- 改 loss 为 GiLt 策略: global/foreg/concat 用 ID loss, parts 用 part-averaged triplet
- 需要先离线提取 PifPaf keypoints 和 body part masks

**Phase 2b (exp004-005)**: 引入 Prompt Tokenizer
- 在 Swin-Tiny 中加 masks_patch_embed
- 零初始化 + keypoint dropping augmentation
- 测试 prompt 对 Occluded-Duke 的增益

**Phase 2c (exp006)**: 精调 + 组合最佳模块
- 调 loss weights, parts_num, dim_reduce 等
- 完整 120 epoch 训练评估

### 关键实现注意事项

1. **Swin-Tiny vs Swin-Base**: KPR 用 Swin-Base (embed_dim=128, depths=(2,2,18,2), 最终 feature_dim=1024)。我们用 Swin-Tiny (embed_dim=96, depths=(2,2,6,2), 最终 feature_dim=768)。所有维度需要相应调整。
2. **显存控制**: KPR 不用 with_cp。我们必须保持 with_cp=True。prompt_tokenizer 和 part attention head 参数量很小不影响。关键是 GWAP 的中间张量 [N, K, D, Hf, Wf]，在 K=8, D=768 时每个 sample ~1.2MB。
3. **Target masks 分辨率**: KPR 将 target_masks 直接 resize 到 backbone 输出的空间分辨率 [Hf, Wf]。我们的 Swin-Tiny 输出 [12, 4] (input 384x128 / 32)。
4. **Shared vs independent part classifiers**: KPR 默认 shared_parts_id_classifier=False，即每个 part 有自己的 BNClassifier。但对于 parts embedding 实际上不用 ID loss (weight=0)，所以这个选择不影响。
5. **Prompt 的 parts grouping 可以与 target masks 的不同**: prompt 用 cck8 (8 body parts from keypoints), target masks 用 pifpaf parsing 的 eight grouping (8 parts from human parsing). 两者粒度一致但来源不同。
