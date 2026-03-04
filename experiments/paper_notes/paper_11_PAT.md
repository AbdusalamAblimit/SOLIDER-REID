# Paper 11: PAT (Part-Aware Transformer)
**来源**: ICCV 2023
**仓库**: https://github.com/liyuke65535/Part-Aware-Transformer
**arXiv 摘要**: 提出 Part-Aware Transformer 用于跨域泛化行人重识别，通过 Part Token + 空间注意力掩码实现局部特征学习，并设计 Cross-ID Similarity Learning (CSL) 和 Part-guided Self-Distillation (PSD) 提升泛化能力。

## 代码架构概览
- 核心文件：`model/backbones/vit_pytorch.py` — ViT backbone 和 Part Attention 实现
- 模型入口：`model/make_model.py` — `build_part_attention_vit` 类
- 训练逻辑：`processor/part_attention_vit_processor.py` — 训练主循环（含 CSL 和 PSD）
- 损失函数：`loss/make_loss.py`（标准 CE+Triplet）, `loss/build_loss.py`（含 PC loss）, `loss/myloss.py`（Pedal, Ipfl 特殊损失）
- 配置文件：`config/PAT.yml`
- 基于 TransReID 框架修改

### 文件结构
```
model/
├── make_model.py                     # 模型构建工厂
├── backbones/
│   ├── vit_pytorch.py                # 核心：part_Attention_ViT, part_Attention, part_Attention_Block
│   ├── resnet.py                     # ResNet 对比实验
│   └── resnet_ibn.py
loss/
├── make_loss.py                      # 标准损失
├── build_loss.py                     # 含 PC loss 的损失
├── myloss.py                         # Pedal + Ipfl 特殊损失
├── triplet_loss.py
├── softmax_loss.py / ce_labelSmooth.py
├── center_loss.py
└── metric_learning.py
processor/
├── part_attention_vit_processor.py   # 训练主循环（核心）
└── ori_vit_processor_with_amp.py     # 普通 ViT 训练
config/
└── PAT.yml
```

## 可拆解模块清单

### 模块 A: Part Token 机制
- 文件位置：`model/backbones/vit_pytorch.py` L564-L666（`part_Attention_ViT.forward_features`）
- 功能：在标准 ViT 的 [CLS] token 之外，引入 3 个额外的 Part Token（part_token1/2/3），分别对应人体的上半身、中间、下半身。这些 Part Token 与 CLS token 和 patch tokens 一起参与 Transformer 的注意力计算，但受到空间注意力掩码的约束，只能关注各自对应的空间区域。
- 输入：image [B, 3, 256, 128]
- 输出：layerwise_tokens — 列表，每个元素 [B, N+4, 768]（N=num_patches, +4=CLS+3parts），共 12 层
- 依赖：无外部依赖
- **移植到我们框架的可行性**：低
- **额外显存开销估算**：~0.3G（3 个额外 token 参与所有层的注意力计算）
- **移植方案**：Swin-Tiny 使用窗口注意力而非全局注意力，无法直接在 patch 序列中插入 Part Token。要移植需要在 Swin 的最后一个 stage（已经是全局注意力或较大窗口）后额外加一个全局注意力层，将 Part Token 作为 query 去聚合 patch 特征。但这违背了 Swin 的高效设计。**不推荐直接移植到 Swin-Tiny**。

### 模块 B: Part Attention Mask（空间注意力掩码）
- 文件位置：`model/backbones/vit_pytorch.py` L88-L118（`generate_2d_mask`）和 L627-L636（`attn_mask_generate`）
- 功能：为 Part Token 生成空间注意力掩码，限制每个 Part Token 只能关注特定的空间区域：
  - Part 1（上半身）：关注图像上半部分 [0, H/2]
  - Part 2（中间）：关注图像中间部分 [H/4, 3H/4]
  - Part 3（下半身）：关注图像下半部分 [H/2, H]
  - CLS token：可以关注所有位置
  - 各 Part Token 之间互不关注（mask[1:4, 0]=0）
  - 每个区域内还有随机裁剪（random crop within the region），增加训练多样性
- 掩码形状：[N+4, N+4] 的 bool 矩阵，应用于每个 attention head
- 输入：H, W（patch grid 尺寸），part index
- 输出：mask [N+4, N+4]
- 依赖：无
- **移植到我们框架的可行性**：高（概念可借鉴，具体实现需要适配）
- **额外显存开销估算**：0（纯掩码，无额外参数）
- **移植方案**：这个空间掩码的思想可以直接应用到我们的 PAMS 模块。我们已经有了基于关键点的 part assignment，PAT 的做法是用固定的空间划分（上/中/下），而我们用关键点驱动的划分更精确。但 PAT 的随机裁剪增强（在区域内随机选子区域）可以借鉴到我们的 part pooling 中，增加 part 特征的鲁棒性。

### 模块 C: part_Attention（带掩码的注意力层）
- 文件位置：`model/backbones/vit_pytorch.py` L202-L232
- 功能：标准 self-attention 的变体，在计算 attention score 后应用掩码：
  1. 正常计算 attn = softmax(Q*K^T / scale)
  2. 对掩码为 0 的位置填充 -1e3（避免 softmax 前的位置被关注）
  3. softmax 后再乘以掩码（确保零化）
  4. 掩码值为 float16 类型（FP16 训练友好）
- 输入：x [B, N+4, 768], mask [B, 1, N+4, N+4]
- 输出：x [B, N+4, 768]
- 依赖：无
- **移植到我们框架的可行性**：低（需要修改 Swin 的注意力机制）
- **额外显存开销估算**：0
- **移植方案**：Swin-Tiny 使用窗口注意力，不适合直接加掩码。但如果我们在 Swin 输出之后加一个额外的 cross-attention 层（Part Tokens 作为 query，patches 作为 key/value），可以在该层使用空间掩码约束 Part Token 的关注区域。这比直接改 Swin 内部更可行。

### 模块 D: Cross-ID Similarity Learning (CSL)
- 文件位置：`processor/part_attention_vit_processor.py` L100-L121 和 `loss/myloss.py` L9-L52（Pedal loss）
- 功能：CSL 是 PAT 的核心创新。其核心思想是：不同 ID 的相似局部部件（如黑色背包、白色鞋子）应该有相似的 part feature。具体实现：
  1. 维护一个 Patch Memory（全局部件特征中心库），每个部件位置存储所有训练样本的特征
  2. 训练时，对每个样本的每个 part token 特征，在 Patch Memory 中找到最近的 K 个邻居
  3. 用 Pedal 损失约束：part token 应该与其 K 个最近邻的距离小于与其他所有中心的距离
  4. K 近邻的 ID 标签会被传递给 ID loss（作为 soft label），实现跨 ID 的局部相似性学习
- 输入：feature [P, B, D]（P=3 parts），centers [P, M, D]（M=memory size），position [B]（分配的中心 ID）
- 输出：loss (scalar), all_posvid（K 近邻的 ID 标签列表）
- 依赖：PatchMemory 全局数据结构（需要额外实现）
- **移植到我们框架的可行性**：中
- **额外显存开销估算**：~0.5G（Patch Memory 存储所有训练样本的 part 特征中心）
- **移植方案**：可以为我们 PAMS 的每个 part 维护一个特征中心库，在训练时做 part-level 的对比学习。这要求在训练开始前先跑一轮前向传播初始化中心（代码 L56-L71 已实现）。对 Occluded-Duke 可能特别有效，因为被遮挡的部位可以从其他样本的相同部位学到一致的特征表示。但实现复杂度较高。

### 模块 E: Part-guided Self-Distillation (PSD)
- 文件位置：`processor/part_attention_vit_processor.py` L116（通过 `soft_label=True` 启用）
- 功能：使用 Part Token 的跨 ID 相似性信息（CSL 中发现的 K 近邻 ID）来生成 soft label，指导 CLS token 的 ID 分类。具体来说：
  1. CSL 找到每个 part 的 K 近邻对应的 ID
  2. 将这些 ID 的频率作为 soft label 分布
  3. 用 soft label + hard label 的加权组合训练 CLS token 的分类器
  - 这使得 CLS token 学到的 global feature 也能关注跨 ID 的局部相似性
- **移植到我们框架的可行性**：中
- **额外显存开销估算**：0（纯计算，依赖 CSL 的结果）
- **移植方案**：如果实现了 CSL（模块 D），PSD 几乎免费获得。用 part 特征的跨 ID 相似性来软化 global ID loss，可能提升泛化能力。

### 模块 F: Layerwise Token 输出
- 文件位置：`model/make_model.py` L287-L298（`build_part_attention_vit.forward`）
- 功能：返回所有层的 token（而非仅最后一层），允许对不同深度的特征进行监督或蒸馏。
  - `layerwise_cls_tokens`：每层的 CLS token，共 12 个 [B, 768]
  - `layerwise_part_tokens`：每层的 3 个 Part Token，12x3 个 [B, 768]
  - 最终只用最后一层的 CLS token 做分类
- **移植到我们框架的可行性**：中
- **额外显存开销估算**：~0.2G（存储所有层的中间特征）
- **移植方案**：可以对 Swin-Tiny 的多个 stage 输出提取 part 特征并做多尺度监督。但我们已经只用最后一个 stage，加多层监督收益未必大。

## 损失函数

1. **ID Loss (CrossEntropyLabelSmooth)**：标签平滑交叉熵，num_classes 通道，epsilon=0.1
   - 可直接用，我们已有

2. **Triplet Loss (SoftTriplet)**：soft margin 三元组损失（无固定 margin）
   - 可直接用，我们已有

3. **Pedal Loss（部件级对比学习）**：
   ```
   loss = sum_over_parts [ log(sum(exp(-scale*neg_dist[:K]))) - log(sum(exp(-scale*neg_dist[:]))) ]
   ```
   - scale=10, K=10（近邻数）
   - 本质是让 K 近邻的概率尽可能大
   - 新颖有趣，但需要 PatchMemory 支持

4. **Ipfl Loss（基于循环排序的三元组）**：
   - 通过 "cycle ranking" 验证负样本的可靠性：对最近的负样本，检查从其出发的 K 近邻中是否包含锚点的 ID，如果不包含才视为可信负样本
   - 最大迭代 15 次寻找可信负样本
   - 实现复杂，效果提升有限

5. **组合方式**：
   ```python
   total_loss = reid_loss + l_ploss * ploss
   # reid_loss = 0.5*ID_LOSS(CLS) + 0.5*mean(ID_LOSS(parts)) + 0.5*TRI_LOSS(CLS) + 0.5*mean(TRI_LOSS(parts))
   # ploss = Pedal loss (CSL)
   # l_ploss 是 PC_LR 配置项
   ```

## 训练 Tricks

- **Backbone**: ViT-Base (768-dim, 12-layer, patch=16x16), ImageNet 预训练
- **输入分辨率**: [256, 128]，stride=[16,16]，patch grid 为 16x8=128 patches
- **优化器**: SGD, lr=0.001, momentum=0.9, weight_decay=1e-4
- **调度**: epoch-based, 60 epochs total
- **BatchSize**: 64 (4 instances per ID)
- **数据增强**:
  - Random Horizontal Flip
  - Random Erasing 未开启（REA.ENABLED: False）
  - Local Grayscale Transformation 未开启
- **AMP 混合精度**: 使用 `amp.autocast` + `GradScaler`（init_scale=512）
- **PatchMemory 初始化**：训练前先跑一轮前向传播，用所有训练样本初始化 part 特征中心
- **评估**：每 epoch 评估，保存最佳 mAP 对应的 checkpoint
- **Part 数量**：固定 3 个（上/中/下），通过空间掩码隐式划分

## 对我们框架的改进建议

1. **空间掩码约束 Part Feature 学习**（优先级：中高）：
   - 核心思想：在我们已有的 PAMS part assignment 基础上，增加空间掩码约束
   - 实现方式：不修改 Swin-Tiny backbone，而是在最后一层 part pooling 时，用关键点位置生成软空间掩码，限制每个 part 只 pool 对应空间区域的特征
   - 优于 PAT 的固定上/中/下划分：我们有关键点信息，可以做更精确的动态划分
   - 显存开销：0

2. **Part-level 对比学习（简化版 CSL）**（优先级：中）：
   - 不需要完整的 PatchMemory 实现
   - 简化方案：对同一 batch 内，相同 part index 但不同 ID 的 part feature 做对比学习
   - 让 "头部" 特征在不同 ID 间保持局部一致性，提升 part feature 的语义一致性
   - 对 Occluded-Duke 可能有帮助：当某个 part 被遮挡时，其他样本的相同 part 可以提供参考
   - 显存开销：<0.1G

3. **随机区域增强**（优先级：低）：
   - PAT 在空间掩码中加入随机裁剪（在区域内随机选子区域）
   - 可以在我们的 part pooling 中也加入类似的随机扰动
   - 但我们已有 Random Erasing，效果可能重叠

4. **不建议移植的模块**：
   - Part Token 机制本身（Swin-Tiny 不兼容全局 Part Token）
   - Ipfl Loss（实现复杂，效果有限）
   - 完整 PatchMemory + Pedal（工程量大，显存开销高）
   - Layerwise 多层监督（我们 Swin-Tiny + with_cp 显存已紧张）
