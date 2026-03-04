# Paper 2: PFD -- Pose-guided Feature Disentangling for Occluded Person Re-identification based on Transformer
**来源**: AAAI 2022
**仓库**: https://github.com/WangTaoAs/PFD_Net
**arXiv 摘要**: PFD uses HRNet pose heatmaps to guide a Transformer encoder-decoder architecture for disentangling pose-relevant features from occluded person images, achieving 69.5% Rank-1 / 61.8% mAP on Occluded-Duke.

## 代码架构概览

### 核心文件
- **主模型定义**: `model/make_pfd.py` -- 包含完整的 `build_skeleton_transformer` 类以及两个关键函数 `PFA` 和 `PVM`
- **姿态估计**: `model/pose_net.py` -- SimpleHRNet 封装，在线调用 HRNet 提取热图
- **HRNet backbone**: `model/hrnet.py` -- HRNet-W48 模型定义
- **ViT backbone**: `model/backbones/vit_pytorch.py` -- 基于 TransReID 的 ViT-Base 实现
- **损失函数**: `loss/make_loss.py` + `loss/pose_push_loss.py` + `loss/triplet_loss.py` + `loss/softmax_loss.py`
- **训练流程**: `processor/processor.py` -- do_train 函数包含完整的训练循环
- **入口**: `occ_train.py`

### 模型入口
`make_pfd(cfg, num_class, camera_num, view_num)` -> 返回 `build_skeleton_transformer` 实例

### 整体流水线 (forward)

```
Input image [bs, 3, 256, 128]
    |
    v
+-- HRNet-W48 (frozen, online inference) --> heatmaps [bs, 17, 64, 32]
|                                             |
|                                             v
|                                        flatten -> [bs, 17, 2048]
|                                             |
|                                     +-------+-------+
|                                     |               |
|                              pose_decoder_linear    pose_avg
|                              FC(2048->768)          AdaptiveAvgPool2d(1,768)
|                              [bs, 17, 768]          [bs, 1, 768]
|                                     |               |
+-- ViT-Base (TransReID) --> features [bs, 129, 768]  |
    |                                                  |
    +-- CLS token * heat_wt + CLS token -> global feat (f_gb)
    |                                                  |
    +-- Patch tokens -> 17 strips (each strip -> b2 -> local_feat_k)
    |   [encoder group features, each 768-dim]         |
    |                                                  |
    +-- features * heat_wt -> decoder_value            |
        |                                              |
        v                                              |
    TransformerDecoder(2 layers) ----query_embed [17, 768]
        |
        v
    last_out [bs, 17, 768]  (decoder part-view features)
        |
        +-- PFA(encoder_local_feats, pose_align_wt) -> sim_decoder [bs, 17, 768]
        |   (Pose Feature Alignment)
        |
        +-- PVM(sim_decoder, last_out) -> decoder_feature [bs, 17, 768]
            (Pose Visibility Matching)
            |
            +-- Split by skeleton visibility:
            |   - visible parts -> skt_parts (high-confidence keypoint features)
            |   - occluded parts -> non_skt_parts (low-confidence features)
            |
            +-- AvgPool -> decoder_out [bs, 768]  (visible part aggregation)
            +-- AvgPool -> non_skt_out [bs, 768]  (occluded part aggregation)

Training returns: encoder_score, encoder_feat, decoder_score, decoder_feat, non_skt_parts
Inference returns: concat(global_feat, 17x encoder_local, decoder_out, 17x decoder_local) = [bs, 768*(1+17+1+17)] = [bs, 27648]
```

## 可拆解模块清单

### 模块 A: Pose Feature Alignment (PFA)
- 文件位置: `model/make_pfd.py` L538-L574
- 功能: 将编码器提取的局部特征与姿态热图引导的特征进行对齐。通过element-wise乘法将姿态信息注入编码器特征，然后用余弦相似度找到最匹配的配对并相加融合。
- 输入:
  - `matrix` [bs, 17, 768]: 编码器的17个局部特征（每个对应一个身体部位strip）
  - `matrix1` [bs, 17, 768]: 姿态引导特征（HRNet heatmap经FC映射到768维）
- 输出: `alignment_feat` [bs, 17, 768]: 姿态对齐后的特征
- 核心逻辑:
  ```python
  # Step 1: 将姿态信息注入到编码器特征
  pose_weighted_feat = matrix * matrix1  # element-wise乘法 [bs, 17, 768]

  # Step 2: 计算原始特征和姿态加权特征之间的余弦相似度
  final_sim = F.cosine_similarity(matrix.unsqueeze(2), pose_weighted_feat.unsqueeze(1), dim=3)  # [bs, 17, 17]

  # Step 3: 找到每个原始特征最匹配的姿态加权特征
  _, ind = torch.max(final_sim, dim=2)  # [bs, 17]

  # Step 4: 将原始特征和匹配的姿态加权特征相加
  new = org_mat[j] + sim_mat[ind[i][j]]  # residual addition
  ```
- 依赖: 无外部依赖，纯tensor操作
- **移植到我们框架的可行性**: 高
- **额外显存开销估算**: 极小（~10MB），仅涉及矩阵运算，无可学习参数
- **移植方案**:
  - 我们的Swin-Tiny输出768维特征，维度完全匹配
  - 关键问题: 需要预先获得17个part特征和17个姿态特征
  - 可直接在我们的PAMS part features上应用，用离线姿态热图替代在线HRNet
  - 注意: 原始实现有大量for循环，需要向量化加速

### 模块 B: Pose Visibility Matching (PVM)
- 文件位置: `model/make_pfd.py` L503-L536
- 功能: 将编码器的姿态对齐特征与解码器的part-view特征进行匹配融合。通过余弦相似度找到最相似的encoder-decoder特征对，然后相加融合。
- 输入:
  - `matrix` [bs, 17, 768]: PFA后的编码器特征
  - `matrix1` [bs, num_query, 768]: 解码器输出的part-view特征
- 输出:
  - `final_feature` [bs, num_query, 768]: 融合后的特征
  - `ind` [bs, num_query]: 匹配索引（用于后续可见性判断）
- 核心逻辑:
  ```python
  # 计算encoder和decoder特征之间的余弦相似度
  final_sim = F.cosine_similarity(matrix.unsqueeze(2), matrix1.unsqueeze(1), dim=3)  # [bs, 17, num_query]
  _, ind = torch.max(final_sim, dim=2)  # 找到每个decoder特征最匹配的encoder特征

  # 将匹配的encoder特征加到decoder特征上（residual fusion）
  new = org_mat[ind[i][j]] + sim_mat[j]
  ```
- 依赖: 无外部依赖
- **移植到我们框架的可行性**: 中
- **额外显存开销估算**: 极小（~10MB），无可学习参数
- **移植方案**:
  - 需要有一个decoder分支来生成part-view特征
  - 在我们的框架中，可以用PAMS的part特征作为encoder侧，再加一个轻量decoder
  - 或简化为: 直接用PFA模块替代，跳过PVM

### 模块 C: Transformer Decoder (Part-View Decoder)
- 文件位置: `model/make_pfd.py` L63-L156
- 功能: 标准的Transformer解码器，使用可学习的query embedding作为part prototypes，通过cross-attention从编码器特征中提取part-view特征
- 输入:
  - `prototype` [num_query, bs, 768]: 初始化为全零的prototype
  - `global_feat` [129, bs, 768]: 编码器输出（经pose热图加权）
  - `query_pos` [num_query, bs, 768]: 可学习的query位置嵌入
- 输出: `output` [num_query, bs, 768]: 解码后的part-view特征
- 关键参数:
  - `d_model=768`, `nhead=8`, `dim_feedforward=2048`, `dropout=0.1`
  - `num_layers=2` (OCC-Duke配置)
  - `num_query=17` (OCC-Duke)
- 依赖: 无外部依赖，标准PyTorch nn.MultiheadAttention
- **移植到我们框架的可行性**: 中
- **额外显存开销估算**: ~50-80MB per decoder layer (2 layers = ~150MB)
  - 每层包含: self-attn(768*768*3), cross-attn(768*768*3), FFN(768*2048*2), 3xLayerNorm
  - 参数量: 约12M per layer, 2 layers = 24M params ~= 96MB (fp32)
- **移植方案**:
  - 可以在Swin-Tiny输出后接一个轻量的2层decoder
  - 输入: Swin-Tiny的patch tokens (经姿态热图加权)
  - query: 5-17个可学习embedding (对应身体部位数)
  - 注意: 我们用with_cp+Swin-Tiny，显存余量有限，decoder层数不宜超过2

### 模块 D: Skeleton Visibility Mask (可见性分离机制)
- 文件位置: `model/make_pfd.py` L265-L411
- 功能: 根据HRNet热图的最大响应值判断每个关键点是否可见（阈值SKT_THRES=0.3），将特征分为可见部位特征(skt_parts)和遮挡部位特征(non_skt_parts)
- 核心逻辑:
  ```python
  # Eq 4: 如果某关键点热图最大值 < threshold，标记为遮挡 (skt_ft=1)
  if max(joint) < self.skeleton_threshold:
      skt_ft[i][j] = 1  # occluded

  # 根据可见性分离decoder特征
  if skt_feat[ind[i][j]] == 1:  # occluded part (通过PVM索引)
      non_skt_feat_list.append(feat)
  else:  # visible part
      per_skt_feat_list.append(feat)

  # 分别聚合
  visible_feat = AdaptiveAvgPool(skt_parts)  # -> [bs, 768]
  occluded_feat = AdaptiveAvgPool(non_skt_parts)  # -> [bs, 768]
  ```
- 依赖: 需要姿态热图的置信度信息
- **移植到我们框架的可行性**: 高
- **额外显存开销估算**: 几乎为0，仅涉及mask操作
- **移植方案**:
  - 我们已有离线提取的关键点置信度
  - 可在PAMS的part特征上直接应用可见性mask
  - 只用可见部位特征计算ID loss和triplet loss
  - 可简化实现: 用关键点置信度作为权重（soft mask），而非硬阈值

### 模块 E: Pose Heatmap Feature Weighting (热图全局加权)
- 文件位置: `model/make_pfd.py` L286-L293
- 功能: 使用姿态热图对编码器特征进行全局性的channel-wise加权
- 核心逻辑:
  ```python
  # 热图平均池化到 [bs, 1, 768] 作为全局姿态权重
  heat_wt = self.pose_avg(heatmaps)  # AdaptiveAvgPool2d: [bs, 17, 2048] -> [bs, 1, 768]

  # 加权方式1: 对CLS token加权（residual）
  feat = features[:, 0].unsqueeze(1) * heat_wt + features[:, 0].unsqueeze(1)

  # 加权方式2: 对所有patch token加权（decoder输入）
  decoder_value = features * heat_wt
  ```
- 依赖: 需要姿态热图
- **移植到我们框架的可行性**: 高
- **额外显存开销估算**: 极小，仅一个AdaptiveAvgPool2d + 一个FC
- **移植方案**:
  - 离线提取17个关键点热图，reshape为 [17, H/4, W/4]
  - 展平+池化到768维，用作channel-wise权重
  - 对Swin-Tiny的输出做 feature * weight + feature（residual weighting）
  - 这是最轻量级的姿态信息注入方式

### 模块 F: Encoder Local Feature Extraction (b2 额外block)
- 文件位置: `model/make_pfd.py` L214-L219, L298-L316
- 功能: 用ViT最后一个block的副本（b2）对每个strip的patch tokens单独处理，提取每个body part的局部特征
- 核心逻辑:
  ```python
  # 复制ViT最后一个block + LayerNorm
  self.b2 = nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm))

  # 将128个patch tokens均分为17个strip
  patch_length = 128 // 17  # = 7 (最后一个strip取剩余)

  # 每个strip + CLS token -> 通过b2 -> 取CLS token输出作为local feature
  local = x[:, patch_length*i:patch_length*(i+1)]
  local_feat = self.b2(torch.cat((token, local), dim=1))
  local_feat_k = local_feat[:, 0]  # CLS token output, [bs, 768]
  ```
- 依赖: ViT最后一个block
- **移植到我们框架的可行性**: 低（Swin-Tiny与ViT架构差异大）
- **额外显存开销估算**: ~40MB (一个Transformer block的参数)
- **移植方案**:
  - Swin-Tiny的输出结构不同，不能直接套用
  - 替代方案: 我们的PAMS已经有part-based pooling，可以用PAMS替代这个模块
  - 或者: 对Swin-Tiny最后一层输出做spatial strip pooling，再过一个小的Transformer block

### 模块 G: Push Loss (姿态推离损失)
- 文件位置: `loss/pose_push_loss.py` L55-L79
- 功能: 将可见部位特征(decoder_out)和遮挡部位特征(non_skt_parts)在特征空间中推远
- 核心逻辑:
  ```python
  # Push_Loss: 计算对角线上的余弦距离（同一样本的visible和occluded特征之间的距离）
  dist = cosine_dist(inputs_1, inputs_2)  # [bs, bs]
  for i, value in enumerate(dist):
      loss += value[i]  # 只取对角线元素
  loss = loss / K
  ```
  这实际上等价于: `loss = mean(cosine_dist(visible_i, occluded_i))`
  目标: 最小化cosine_dist = 最大化两组特征的差异性（因为cosine_dist = (1-cos)/2，越小表示越相似）

  等等，仔细看代码，cosine_dist 返回 (1-cos)/2，范围 [0,1]。loss = mean(cosine_dist)，是在最小化这个距离。这意味着模型在训练时是让 visible 和 occluded 特征更相似？

  **重新分析**: 查看 `processor.py` L77: `loss_push = push_single_loss(out[0], non_skt_parts)`
  - `out[0]` = decoder_out (经BN的可见部位特征)
  - `non_skt_parts` = 遮挡部位特征
  - cosine_dist = (1 - cos_similarity) / 2, 越小表示越相似
  - 最小化 cosine_dist 意味着让 visible 特征和 occluded 特征更相近？

  **实际上这是有道理的**: push loss 的目标是让同一个人的可见部位特征能够"补偿"遮挡部位的特征。即: 即使某些部位被遮挡，模型也应该从可见部位推断出一个和完整特征一致的表示。这不是"推远"而是"拉近"可见和遮挡的特征。

  **注意**: 但命名是 "Push_Loss"，可能是原始论文中描述的"推离不同身份的遮挡特征"的简化版本。需要结合论文确认。在代码中，loss_push的权重为1.0（直接加到total loss），且使用的是同一个identity的visible和occluded特征。
- 依赖: 需要可见性分离后的两组特征
- **移植到我们框架的可行性**: 高
- **额外显存开销估算**: 0（无参数）
- **移植方案**:
  - 直接在我们的PAMS part特征上实现
  - 根据关键点置信度分出可见/遮挡特征
  - 加一个cosine distance loss让同一身份的可见/遮挡表示一致

## 损失函数

### 1. Encoder Loss
```
L_encoder = 0.5 * ID_loss + 0.5 * Triplet_loss
```
- **ID Loss**: Label-smoothed Cross-Entropy (epsilon=0.1)
  - 对encoder全局特征: 0.5权重
  - 对17个encoder局部特征: 均值 * 0.5权重
  - 公式: `0.5 * xent(global_score, target) + 0.5 * mean([xent(local_score_k, target) for k in 1..17])`
- **Triplet Loss**: Soft margin triplet loss (NO_MARGIN=True, 使用SoftMarginLoss)
  - 同样分全局和局部: `0.5 * triplet(global_feat) + 0.5 * mean([triplet(local_feat_k) for k in 1..17])`
  - 使用 euclidean distance + hard example mining
- 权重: `ID_LOSS_WEIGHT=1.0`, `TRIPLET_LOSS_WEIGHT=1.0`

### 2. Decoder Loss
```
L_decoder = 0.5 * ID_loss + 0.5 * Triplet_loss
```
- 结构与encoder loss完全相同，但作用于decoder的特征:
  - decoder全局特征 (decoder_out)
  - decoder的num_query个局部特征

### 3. Push Loss
```
L_push = Push_Loss(decoder_out, non_skt_parts)
```
- cosine距离（同一样本的可见特征和遮挡特征之间）
- 权重: 1.0（直接加到total loss）

### 4. Total Loss
```
L_total = 0.5 * L_encoder + 0.5 * L_decoder + L_push
```

### 可直接使用的损失
- **Push Loss**: 可直接移植，概念简单且无额外参数
- **分层ID+Triplet**: 我们已经有类似的loss结构（global + part），可以参考其0.5/0.5的权重配比
- **Soft Margin Triplet**: 比hard margin更稳定，我们可以切换

## 训练 Tricks

### 1. 超参数配置 (Occluded-Duke)
| 参数 | 值 | 备注 |
|------|-----|------|
| Backbone | ViT-Base (768-dim, 12 heads, 12 layers) | 我们用 Swin-Tiny (768-dim) |
| 输入尺寸 | 256x128 | 我们用 384x128 |
| Batch Size | 64 (16 IDs x 4 instances) | - |
| Optimizer | SGD | momentum=0.9 |
| Base LR | 0.008 | 对于SGD来说偏高 |
| LR Schedule | Cosine Annealing | lr_min=0.002*base_lr, warmup=5 epochs |
| Warmup | Linear, 5 epochs | warmup_lr_init=0.01*base_lr |
| Weight Decay | 1e-4 | - |
| Epochs | 300 | - |
| Label Smoothing | 0.1 | - |
| SIE (Camera Embedding) | 开启，系数3.0 | 相机感知 |
| Stride Size | [16, 16] | 非overlap patch |
| Decoder Layers | 2 | Market用6层，OCC-Duke用2层 |
| Decoder Heads | 8 | - |
| Decoder Dropout | 0.1 | - |
| Query Num | 17 | 与关键点数一致 |
| SKT Threshold | 0.3 | 可见性判断阈值 |
| Pixel Norm | mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5] | ViT标准 |
| AMP | 开启 (amp.autocast + GradScaler) | - |

### 2. 数据增强
- Random Horizontal Flip (p=0.5)
- Random Padding (10px) + Random Crop
- Random Erasing (p=0.5, pixel mode)

### 3. HRNet 配置
- HRNet-W48 (c=48, 17 joints)
- 输入分辨率: 256x128 (与ReID模型相同)
- 热图输出: [bs, 17, 64, 32] (1/4分辨率)
- **在线推理** (frozen, no_grad) -- 重要: 这会占用额外显存!
- 可见性阈值: 0.3 (热图最大值 < 0.3 判定为遮挡)

### 4. 特征维度设计
- ViT特征: 768-dim
- 姿态热图展平: 2048-dim (64*32=2048)
- FC映射: 2048 -> 768 (唯一的新增FC)
- 每个part一个独立的classifier (768 -> num_classes)
- 每个part一个独立的BN (BatchNorm1d)

### 5. 推理特征
- 将encoder全局、17个encoder局部、decoder全局、17个decoder局部全部concat
- 总维度: 768 * (1 + 17 + 1 + 17) = 768 * 36 = 27,648 维
- 使用euclidean distance计算距离矩阵
- 测试特征L2归一化: 开启

### 6. 注意事项
- 代码使用大量`exec()`动态创建变量，代码质量不高但功能正确
- HRNet在线推理时需要CPU/GPU之间频繁转换（`.cpu().numpy()` / `.cuda()`），效率低
- PFA和PVM中的for循环是主要的速度瓶颈

## 关键设计洞察

### 1. 双分支设计 (Encoder + Decoder)
- Encoder分支: 通过strip partition生成局部特征 (类似TransReID的JPM)
- Decoder分支: 通过cross-attention从编码器特征中提取part-view特征
- 两个分支的特征通过PFA和PVM进行对齐和融合
- 最终推理时concat两个分支的所有特征

### 2. 姿态信息的三种使用方式
1. **全局加权**: 热图均值作为channel-wise权重 (heat_wt)
2. **特征对齐**: 热图经FC映射后与encoder局部特征进行PFA对齐
3. **可见性判断**: 热图最大值用于判断关键点可见性，分离遮挡/非遮挡特征

### 3. 姿态-特征对齐的关键步骤
- 步骤1: HRNet热图 [bs,17,64,32] -> 展平 [bs,17,2048] -> FC [bs,17,768]
- 步骤2: ViT patch tokens均分17 strips -> 通过额外block b2 -> 17个local features [bs,17,768]
- 步骤3: PFA: 将步骤1和步骤2的特征通过余弦相似度进行最优匹配对齐
- 步骤4: PVM: 将PFA后的特征与decoder part-view特征进行匹配融合

## 对我们框架的改进建议

### 建议1: 离线姿态热图 + Soft Visibility Weighting (优先级: 最高)
- **做法**: 离线用HRNet提取17个关键点热图，存储为 [17, H/4, W/4]
- 在PAMS的part特征上应用soft visibility weighting:
  ```python
  # 每个part的visibility score = 对应关键点热图的最大值
  visibility = heatmap.max(dim=-1).max(dim=-1)  # [bs, 17]
  # 软加权
  weighted_part_feat = part_feat * visibility.unsqueeze(-1)
  ```
- 推理时只聚合高置信度的part特征
- **预期收益**: 直接提升遮挡场景的鲁棒性
- **显存成本**: 几乎为0

### 建议2: Pose Feature Alignment Module (优先级: 高)
- **做法**: 将PFA模块简化后接入我们的PAMS
- 离线热图 -> FC映射到768维 -> 与PAMS part特征做余弦匹配 + residual相加
- **关键简化**:
  - 不需要decoder分支和PVM，只保留PFA
  - 将for循环向量化
  - 可将17个关键点合并为5个部位（head, torso, left-arm, right-arm, legs），减少计算
- **预期收益**: mAP +1-2%
- **显存成本**: ~20MB (一个FC层)

### 建议3: Push Loss (可见-遮挡特征一致性) (优先级: 高)
- **做法**: 基于关键点置信度将part特征分为可见和遮挡两组
- 对同一identity的可见/遮挡聚合特征施加cosine一致性loss
- 目标: 让模型从可见部位推断出与完整特征一致的表示
- **预期收益**: mAP +0.5-1% (在严重遮挡场景下更明显)
- **显存成本**: 0

### 建议4: 轻量 Part-View Decoder (优先级: 中)
- **做法**: 在Swin-Tiny输出后接一个1-2层的Transformer decoder
- 使用5个可学习query (对应5个body parts)
- cross-attention从Swin-Tiny的patch tokens中提取part特征
- **注意**: 这会增加~80-150MB显存，需要评估是否可行
- **预期收益**: mAP +1-3% (如果decoder能学到好的part分解)
- **显存成本**: ~100-150MB

### 建议5: 不建议移植的部分
- **在线HRNet推理**: 太占显存（HRNet-W48约500MB），必须用离线方式
- **b2额外block**: 针对ViT的设计，不适合Swin-Tiny
- **17个独立classifier和BN**: 参数量大且容易过拟合，我们的PAMS共享classifier可能更好
- **原始PVM**: 需要decoder分支配合，单独使用无意义

### 建议6: 实现优先级排序
1. 离线提取关键点 + soft visibility weighting (exp: +0.5-1.5% mAP, 无显存增加)
2. PFA简化版 + push loss (exp: +1-2% mAP, ~20MB额外)
3. 轻量decoder (exp: +1-3% mAP, ~100-150MB额外，需评估显存)

### 与我们框架的关键差异
| 方面 | PFD | 我们的框架 |
|------|-----|-----------|
| Backbone | ViT-Base (768-dim, 86M params) | Swin-Tiny (768-dim, 28M params) |
| 姿态模型 | HRNet-W48 (在线) | 需离线提取 |
| Part分解 | Strip partition (17 strips) | PAMS (part-aware, attention-based) |
| 输入尺寸 | 256x128 | 384x128 |
| Decoder | 2层 Transformer Decoder | 无 (可加) |
| 可见性 | 硬阈值0.3 | 待实现 (建议soft weighting) |
| 推理特征维度 | 27,648 | 待确认 |
| Occluded-Duke mAP | 60.1% (无slide) / 61.8% (有slide) | 59.0% (当前最佳) |
