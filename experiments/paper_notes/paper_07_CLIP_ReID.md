# Paper 7: CLIP-ReID -- Exploiting Vision-Language Model for Image Re-Identification without Concrete Text Labels
**来源**: AAAI 2023
**仓库**: https://github.com/Syliz517/CLIP-ReID
**arXiv 摘要**: 提出利用 CLIP 预训练的视觉-语言模型进行 ReID，通过两阶段训练（先学习 text prompt 再微调视觉编码器）将 CLIP 的跨模态对齐能力迁移到 ReID 任务，无需人工设计文本标签即可获得强大的身份区分特征。

## 代码架构概览
- 核心模型文件：`model/make_model_clipreid.py`（CLIP-ReID 模型定义，包含 `PromptLearner`、`TextEncoder`、`build_transformer`）
- CLIP 模型：`model/clip/model.py`（修改版 CLIP，支持可变分辨率和 stride_size，返回多层特征）
- 训练入口：`train_clipreid.py`（两阶段训练流程）
- 训练 Stage 1：`processor/processor_clipreid_stage1.py`（文本 prompt 学习）
- 训练 Stage 2：`processor/processor_clipreid_stage2.py`（视觉+文本联合微调）
- 损失函数：`loss/make_loss.py`（标准 ID + triplet + I2T loss），`loss/supcontrast.py`（SupConLoss 对比学习）
- 配置：`configs/person/vit_clipreid.yml`

训练为严格的两阶段：
- Stage 1：冻结视觉编码器和文本编码器的主体，只训练 `PromptLearner` 的可学习 token，通过 image-text 对比损失对齐每个 ID 的文本描述
- Stage 2：冻结 `PromptLearner` 和 `TextEncoder`，训练视觉编码器 + BNNeck + classifiers，使用 ID loss + triplet loss + I2T loss

## 可拆解模块清单

### 模块 A: PromptLearner -- 可学习的文本 Prompt
- 文件位置：`model/make_model_clipreid.py` L191-L238
- 功能：为每个身份 ID 学习一组可微分的文本 token embedding（类似 CoOp 的思路），替代手工设计的文本描述。每个 ID 有独立的 4 个可学习 context token，嵌入到模板 "A photo of a [X X X X] person." 中。
- 输入：`label` [B]，身份 ID
- 输出：`prompts` [B, seq_len, ctx_dim]，可送入文本编码器
- 依赖：需要 CLIP 的 token_embedding 层进行初始化
- 实现细节：
  - 模板：`"A photo of a X X X X person."`（person 数据集）或 `"A photo of a X X X X vehicle."`（vehicle 数据集）
  - `ctx_dim = 512`（CLIP 文本 transformer 宽度）
  - `n_cls_ctx = 4`：每个 ID 有 4 个可学习 token
  - `cls_ctx = nn.Parameter(torch.empty(num_class, n_cls_ctx, ctx_dim))`，用 `std=0.02` 的正态分布初始化
  - 前向传播：根据 label 索引对应 ID 的 ctx tokens，与 prefix/suffix 拼接成完整 prompt
  - `token_prefix`：模板前半部分 "A photo of a" 的 embedding（frozen buffer）
  - `token_suffix`：模板后半部分 "person." 的 embedding（frozen buffer）
  - 总参数量 = num_class x 4 x 512（Occluded-Duke 约 702 个 ID，即 702x4x512 = 1.44M 参数）
- **移植到我们框架的可行性**：低
  - 我们使用 Swin-Tiny（纯视觉模型），没有文本编码器
  - PromptLearner 的核心价值在于利用 CLIP 的视觉-文本对齐，脱离 CLIP 后无意义
  - 但其 "per-ID learnable embedding" 的思想可以借鉴：为每个 ID 学习一个可查找的 prototype embedding，用于辅助训练（类似 center loss 但可微分）
- **额外显存开销估算**：约 0.01G（参数量很小，1.44M float32 约 5.5MB）
- **移植方案**：不直接移植。但思想可借鉴为 "ID-conditioned prototype" 方式增强特征学习

### 模块 B: TextEncoder -- CLIP 文本编码器
- 文件位置：`model/make_model_clipreid.py` L31-L50
- 功能：将 PromptLearner 生成的 prompt embedding 编码为文本特征向量
- 输入：`prompts` [B, seq_len, dim]，`tokenized_prompts` [1, seq_len]
- 输出：`text_features` [B, 512]
- 实现细节：
  - 使用 CLIP 的 Transformer 编码器（12 层, 512-dim, 8 heads）
  - 取 EOT (end of text) token 的输出经过 text_projection 投影为 512 维
  - Stage 1 时全量可用，Stage 2 时冻结
- **移植到我们框架的可行性**：低（需要完整的 CLIP 文本编码器，显存开销大）
- **额外显存开销估算**：约 0.3G（CLIP 文本 Transformer 约 63M 参数）
- **移植方案**：不推荐。除非我们改用 CLIP-ViT 作为 backbone

### 模块 C: I2T Loss -- Image-to-Text 对比损失
- 文件位置：`loss/make_loss.py` L54-L56（I2T loss 入口），`loss/supcontrast.py` L10-L29（SupConLoss 实现）
- 功能：计算图像特征与文本特征之间的监督对比损失，拉近同 ID 的图文对，推远不同 ID 的图文对
- 输入：
  - `image_features` [B, D]：图像编码器输出的投影特征
  - `text_features` [num_class, D]：所有 ID 的文本 prototype（Stage 2 中预计算并缓存）
- 输出：标量 loss
- 实现细节：
  - `logits = image_features @ text_features.T / temperature`（temperature=1.0）
  - 用 label mask 识别正样本对，计算 log_prob 的加权平均
  - 数值稳定：`logits = logits - logits_max.detach()`
  - Stage 2 loss = `ID_LOSS_WEIGHT * ID_loss + TRIPLET_LOSS_WEIGHT * Triplet_loss + I2T_LOSS_WEIGHT * I2T_loss`
  - 默认权重：ID=0.25, Triplet=1.0, I2T=1.0
  - Stage 1 loss = `SupConLoss(img->text) + SupConLoss(text->img)`，双向对比
- **移植到我们框架的可行性**：中
  - I2T loss 的核心思想是将图像特征与 "每个 ID 的 prototype" 做对比
  - 不使用 CLIP 时，可以用 "ID center features"（如 moving average 的 class center）替代 text features
  - 这就变成了一种 "Image-to-Prototype Contrastive Loss"，与 center loss 有相似之处但使用 softmax 对比而非 L2 距离
- **额外显存开销估算**：约 0.01-0.1G（取决于 num_class x D 的 prototype 矩阵大小）
- **移植方案**：
  1. 维护一个 `[num_class, D]` 的 class prototype matrix（EMA 更新或可学习参数）
  2. 每个 batch 计算 image features 与所有 prototypes 的相似度 logits
  3. 使用 SupConLoss 或简单的 CE loss 作为辅助监督
  4. 这相当于在 ID loss 基础上增加一个 prototype 对齐的正则化项

### 模块 D: 两阶段训练策略
- 文件位置：`train_clipreid.py` L66-L100，`processor/processor_clipreid_stage1.py`，`processor/processor_clipreid_stage2.py`
- 功能：先学 prompt，再微调视觉
- Stage 1 细节：
  - 120 epochs，Adam，lr=3.5e-4，cosine schedule
  - 只更新 `PromptLearner` 参数
  - 先用 frozen 视觉编码器提取所有训练图像的特征（离线，一次性）
  - 然后用 SupConLoss(i2t) + SupConLoss(t2i) 双向对比训练 prompt
  - 使用 AMP (fp16) 加速
- Stage 2 细节：
  - 60 epochs，Adam，lr=5e-6（非常小的 LR）
  - 冻结 PromptLearner 和 TextEncoder
  - 先预计算所有 ID 的 text features（离线，一次性）
  - 训练视觉编码器 + BNNeck + classifiers
  - Loss = 0.25 * ID + 1.0 * Triplet + 1.0 * I2T(logits = img_feat @ text_feat.T)
  - MultiStepLR: steps=[30, 50], gamma=0.1
  - Warmup: 10 iterations, factor=0.1
- **移植到我们框架的可行性**：中（策略层面）
  - 两阶段训练的思想可以借鉴：先用辅助任务预热部分参数，再做主任务微调
  - 例如：先用 pose estimation 辅助任务预训练 pose-aware 模块，再做 ReID 微调
  - 但在我们的场景下，一阶段端到端训练更简单高效
- **移植方案**：不直接移植两阶段流程，但可以借鉴 "辅助目标预热 + 主目标微调" 的思想

### 模块 E: 修改版 CLIP VisionTransformer（多层特征返回）
- 文件位置：`model/clip/model.py` L200-L240
- 功能：修改 CLIP ViT 使其返回三层特征：第 11 层输出（x11）、第 12 层输出（x12）、投影特征（xproj）
- 实现细节：
  - `x11 = self.transformer.resblocks[:11](x)`
  - `x12 = self.transformer.resblocks[11](x11)`
  - `xproj = x12 @ self.proj` (768->512 投影)
  - SIE: `cv_embed` 加到 cls_token 上（`x[:,0] = x[:,0] + cv_emb`）
  - 位置编码 resize: 从 14x14 双线性插值到 h_resolution x w_resolution
  - 支持 stride_size 参数控制 patch 密度
- **移植到我们框架的可行性**：中（多层特征返回的思想）
  - Swin-Tiny 已经天然支持多 stage 特征（96/192/384/768 dim）
  - 可以利用中间 stage 的特征做辅助损失或多尺度融合
  - 我们的 PAMS 已经在多 stage 上操作
- **移植方案**：中间层特征利用已在 PAMS 中实现

### 模块 F: SIE on CLS Token Only
- 文件位置：`model/clip/model.py` L218-L224, `model/make_model_clipreid.py` L91-L101, L127-L138
- 功能：CLIP-ReID 的 camera/view embedding 仅加在 cls_token 上（而非所有 token）
- 实现细节：
  - `cv_embed = nn.Parameter(torch.zeros(camera_num, in_planes))`，in_planes=768
  - 视觉编码器中：`x[:,0] = x[:,0] + cv_emb`（仅修改 cls_token）
  - 这与 TransReID 的做法不同（TransReID 加到所有 token 的 pos_embed 上）
- **移植到我们框架的可行性**：中
  - Swin 没有 cls_token，但可以在全局 average pooling 后的特征上加 camera embedding
  - 这种 "仅在全局特征上加 SIE" 的做法更轻量
- **移植方案**：在 Swin 输出的全局特征（GAP 后）上直接加 camera embedding

## 损失函数

### ID Loss (Cross Entropy Label Smooth)
- epsilon=0.1 的标准 label smoothing
- 权重：0.25（显著低于 triplet 和 I2T）
- 双路分类器：classifier (768-dim) + classifier_proj (512-dim)
- 可否直接用：我们已有类似实现

### Triplet Loss (Soft Margin, Hard Mining)
- 与 TransReID 完全一致的实现
- 权重：1.0
- 可否直接用：已有

### I2T Loss (Image-to-Text Supervised Contrastive)
- `SupConLoss`: 双向监督对比，temperature=1.0
- 用 image_features @ text_features.T 计算 logits
- 可否直接用：需要 text features 或 prototype matrix 才能使用

### 损失组合
```
L_total = 0.25 * L_id + 1.0 * L_triplet + 1.0 * L_i2t
```
注意 ID loss 权重很低 (0.25)，说明在有 I2T 对比损失的情况下，ID loss 的重要性降低。

## 训练 Tricks

### 超参数
- Stage 1: Adam, lr=3.5e-4, 120 epochs, cosine schedule
- Stage 2: Adam, lr=5e-6, 60 epochs, MultiStepLR [30,50] gamma=0.1
- Stage 2 LR 极低（5e-6），因为 CLIP backbone 已经很强，只需微调
- Warmup: Stage 1 有 5 epoch warmup (lr_init=1e-5); Stage 2 有 10 iter warmup (factor=0.1)
- 使用 AMP (fp16) 加速

### 数据增强
- 与 TransReID 基本一致：resize + random flip + padding + random crop + random erasing
- Pixel mean/std: [0.5, 0.5, 0.5]

### 关键设计决策
1. **Stage 1 离线提取图像特征**：避免重复前向传播，大幅加速 prompt 学习
2. **Stage 2 离线预计算 text features**：将 [num_class, 512] 的文本 prototype 缓存到 GPU，训练时直接做矩阵乘
3. **双 BNNeck 双分类器**：一个处理 768-dim 的原始特征，一个处理 512-dim 的投影特征
4. **冻结策略**：Stage 1 只训练 PromptLearner（约 1.4M 参数），Stage 2 冻结 text 侧只训练 vision 侧
5. **测试时特征拼接**：`torch.cat([feat, feat_proj], dim=1)` -> 768+512=1280 维

### 报告性能
| 方法 | 数据集 | mAP | Rank-1 |
|------|--------|-----|--------|
| ViT-baseline | OCC-Duke | - | - |
| ViT-CLIP-ReID | OCC-Duke | 有模型链接但未在表中列出具体数字 | - |
| ViT-CLIP-ReID+SIE+OLP | Market | 89.8 | 95.7 |
| ViT-CLIP-ReID+SIE+OLP | MSMT17 | 75.1 | 89.8 |

## 对我们框架的改进建议

1. **ID Prototype Contrastive Loss（优先级：高，核心可借鉴思想）**
   - CLIP-ReID 的 I2T loss 本质上是让图像特征与 "每个 ID 的 prototype 特征" 对齐
   - 不需要 CLIP 文本编码器，我们可以用可学习的 prototype matrix `[num_class, D]` 替代
   - 实现：`logits = L2_normalize(image_feat) @ L2_normalize(prototype_matrix).T`，然后用 CE loss
   - 这类似于 CosFace/ArcFace 分类器，但不同之处在于：(1) 特征维度可以比分类器维度低 (2) 可以用 EMA 更新 prototype 而非纯可学习
   - 预期增益：增强全局特征的判别力，尤其对遮挡场景有帮助
   - 显存开销：极小（num_class x D 的矩阵，约 702 x 768 x 4B = 2.2MB）

2. **低 ID Loss 权重 + 强对比损失（优先级：中）**
   - CLIP-ReID 用 0.25 的 ID loss 权重，远低于 triplet (1.0) 和 I2T (1.0)
   - 这暗示在有强对比损失的情况下，ID 分类损失可以适当降低权重
   - 我们可以在 PAMS 框架下尝试：降低 ID loss 权重（从 1.0 到 0.5），增加 triplet loss 权重或增加 prototype contrastive loss

3. **Stage 2 极低 LR 微调策略（优先级：低）**
   - CLIP-ReID Stage 2 只用 5e-6 的 LR，说明 CLIP 预训练已经提供了很好的初始化
   - 类比到我们：SOLIDER 预训练的 Swin-Tiny 也有很好的初始化，我们或许可以尝试更低的 backbone LR + 更高的 head LR
   - 差异学习率策略可以进一步探索

4. **双特征空间拼接（优先级：低）**
   - CLIP-ReID 在推理时拼接原始特征 (768d) 和投影特征 (512d)
   - 我们可以类比：拼接 PAMS 的全局特征和 part features 用于检索
   - 我们目前的 PAMS 评估已经使用 global + part 联合评分，已经覆盖此思路

5. **不推荐移植的部分**
   - PromptLearner：依赖 CLIP 文本编码器，无法移植到纯视觉模型
   - TextEncoder：同上
   - 完整两阶段训练流程：增加训练复杂度，一阶段端到端训练更适合我们的场景
   - CLIP backbone 替换：我们的 Swin-Tiny + SOLIDER 预训练已经是高效选择，替换为 CLIP-ViT-B/16 会显著增加计算量且不一定更好（CLIP 未针对人体做预训练）
