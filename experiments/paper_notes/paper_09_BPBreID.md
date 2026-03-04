# Paper 9: BPBreID -- Body Part-Based Representation Learning for Occluded Person Re-Identification
**来源**: WACV 2023
**仓库**: https://github.com/VlSomers/bpbreid
**arXiv 摘要**: BPBreID proposes a part-based ReID method that learns an attention mechanism to pool spatial features into K body-part embeddings using pseudo human parsing labels at training time; at test time, only mutually visible parts are compared, achieving SOTA on occluded ReID benchmarks.

## 代码架构概览
- 核心文件：
  - `torchreid/models/bpbreid.py` -- 主模型定义（BPBreID class）
  - `torchreid/engine/image/part_based_engine.py` -- 训练/测试引擎（ImagePartBasedEngine）
  - `torchreid/losses/GiLt_loss.py` -- GiLt (Global-identity Local-triplet) 复合损失
  - `torchreid/losses/body_part_attention_loss.py` -- 像素级部件分类损失
  - `torchreid/losses/part_averaged_triplet_loss.py` -- 部件平均三元组损失
  - `torchreid/metrics/distance.py` -- 基于 part 的距离矩阵计算 + 可见性感知
  - `torchreid/data/masks_transforms/pifpaf_mask_transform.py` -- PifPaf 关键点到部件 mask 的分组
  - `torchreid/data/masks_transforms/mask_transform.py` -- mask 分组/背景计算/resize 变换
- 模型入口：`BPBreID.forward()` in `bpbreid.py` L116-L259
- 基于 Torchreid 框架改造，添加了完整的 part-based 训练/评估流水线
- 默认 backbone：HRNet-W32（也支持 ResNet-50、ResNet-IBN 等）

### 核心设计哲学
BPBreID 的核心论点：**全局嵌入在局部可观测性下有理论缺陷**。当两个图像只展示同一个人的不同部分（如上半身 vs 下半身）时，基于全局嵌入的方法无法正确处理。Part-based 方法通过只比较共同可见的部件来解决这个问题。

## 可拆解模块清单

### 模块 A: PixelToPartClassifier（像素到部件分类器）
- 文件位置：`torchreid/models/bpbreid.py` L376-L395
- 功能：将 backbone 输出的 spatial feature map 中每个像素分类到 K+1 个类别（K 个部件 + 1 个背景），学习出 learnable attention maps
- 结构：`BN2d(D) -> Conv2d(D, K+1, 1x1)`
- 输入：spatial features `[N, D, Hf, Wf]`（如 D=2048 for HRNet-32, Hf=24, Wf=8 for 384x128 input）
- 输出：像素分类 logits `[N, K+1, Hf, Wf]`，经 softmax 后得到 per-pixel part probabilities
- 依赖：无
- **移植到我们框架的可行性**：**高**
- **额外显存开销估算**：极小（一个 1x1 Conv），约 `D*(K+1)*4 bytes`。对于 D=768, K=5: 约 18K params = ~0.07MB
- **移植方案**：直接在我们的 PAMS backbone 输出的 spatial feature map 上加一个 `BN2d(768) -> Conv2d(768, K+1, 1x1)` 即可。输出的 attention maps 可以替代或辅助我们现有的基于关键点的 part 划分。相比我们用关键点硬性划分，这种 learnable attention 更灵活，可以自适应地学习部件边界。但需要提供训练时的 pseudo parsing labels。

**关键洞察**：我们的 PAMS 已经使用了类似的思路（5-part attention），但 PAMS 的 part attention 是基于 pose keypoint 关键点坐标硬性生成的 Gaussian heatmap，而 BPBreID 的是通过像素分类器学习出来的。可以考虑两者结合：用 pose heatmap 作为监督信号来训练 learnable attention。

### 模块 B: Part-based Attention Pooling (部件注意力池化)
- 文件位置：`torchreid/models/bpbreid.py` L432-L503
- 功能：使用 attention masks（来自 PixelToPartClassifier 或外部 parsing labels）对 spatial feature map 进行加权池化，生成 K 个部件嵌入
- 三种池化策略：
  1. **GAP (GlobalAveragePoolingHead)**：`features * masks -> AdaptiveAvgPool2d(1)` -- 对 mask 加权后的特征做全局平均池化
  2. **GMP (GlobalMaxPoolingHead)**：`features * masks -> AdaptiveMaxPool2d(1)` -- 对 mask 加权后的特征做全局最大池化
  3. **GWAP (GlobalWeightedAveragePoolingHead)** [默认/最佳]：`sum(features * masks) / sum(masks)` -- 真正的加权平均，按 mask 值归一化
- 输入：spatial features `[N, D, Hf, Wf]` + part masks `[N, K, Hf, Wf]`
- 输出：part embeddings `[N, K, D]`
- 依赖：无
- **移植到我们框架的可行性**：**高**（与 PAMS 直接对应）
- **额外显存开销估算**：极小（无可学习参数，仅计算操作）
- **移植方案**：我们 PAMS 的 part pooling 已经是类似的 attention-weighted pooling。BPBreID 的 GWAP 与我们的实现基本等价。关键区别在于 mask 的来源：BPBreID 用 learnable softmax attention，我们用 keypoint-based Gaussian heatmaps。

### 模块 C: GiLt Loss (Global-identity Local-triplet 损失策略)
- 文件位置：`torchreid/losses/GiLt_loss.py` L1-L119
- 功能：一个灵活的复合损失框架，对不同级别的嵌入（global、foreground、concat_parts、parts）独立配置 ID loss 和 triplet loss 的权重
- 默认 GiLt 策略：
  ```
  global:      ID=1.0, triplet=0.0   (全局嵌入只用 ID loss)
  foreground:  ID=1.0, triplet=0.0   (前景嵌入只用 ID loss)
  concat_parts: ID=1.0, triplet=0.0  (拼接部件嵌入只用 ID loss)
  parts:       ID=0.0, triplet=1.0   (各部件嵌入只用 triplet loss)
  ```
- 核心思想：**全局特征用 ID loss 学习身份判别，局部部件特征用 triplet loss 学习细粒度匹配**。这避免了对每个部件单独做 ID 分类（部件可能不够判别），也避免了对全局特征做 triplet（全局 triplet 容易被遮挡干扰）
- 输入：embeddings_dict, visibility_scores_dict, id_cls_scores_dict, pids
- 输出：加权总 loss + loss_summary
- **移植到我们框架的可行性**：**中** -- 我们已有 ID loss + triplet loss，但缺乏 GiLt 的灵活配置
- **额外显存开销估算**：0（纯 loss 计算逻辑）
- **移植方案**：
  1. 我们当前 PAMS 已经有 global ID loss + per-part ID loss + triplet loss
  2. 可以借鉴 GiLt 的思路：保留 global/foreground 的 ID loss，但将 per-part 的 loss 从 ID loss 改为 triplet loss
  3. 这需要实验验证：对于 Swin-Tiny 768-dim 的 part features，triplet loss 是否比 ID loss 更有效？
  4. **关键问题**：我们的 part features 经过了降维（768 -> part_dim），维度可能不够做有效的 triplet mining

### 模块 D: Body Part Attention Loss (像素级部件注意力损失)
- 文件位置：`torchreid/losses/body_part_attention_loss.py` L1-L52
- 功能：用 CrossEntropyLoss（带 label smoothing）训练 PixelToPartClassifier，使其学习正确的部件分割
- 输入：pixels_cls_scores `[N, K+1, Hf, Wf]`（from PixelToPartClassifier），target parsing labels `[N, Hf, Wf]`（from external masks, argmax 后的整数标签）
- 输出：像素分类 loss + accuracy
- 支持三种 loss 类型：CrossEntropy (cl, 默认), FocalLoss (fl), DiceLoss (dl)
- 权重：在 Occ-Duke 配置中为 0.35
- **移植到我们框架的可行性**：**高** -- 如果我们加入 learnable attention，需要对应的监督损失
- **额外显存开销估算**：0（纯 loss 计算）
- **移植方案**：如果我们引入 PixelToPartClassifier，就需要同步引入此 loss。需要为我们的数据集准备 human parsing pseudo labels。可以利用已有的 pose keypoint 信息生成粗略的部件 parsing labels。

### 模块 E: Part-Averaged Triplet Loss (部件平均三元组损失)
- 文件位置：`torchreid/losses/part_averaged_triplet_loss.py` L1-L225
- 功能：将 K 个部件的 pairwise distance matrix 先平均合并为一个 sample-to-sample distance matrix，然后做标准 batch-hard triplet mining
- 核心流程：
  1. 计算 part-based pairwise distances `[K, N, N]`
  2. 可选：用 visibility scores 生成 valid distance mask
  3. 合并策略：`masked_mean` -- 对所有互相可见的部件距离取平均
  4. 标准 batch-hard triplet mining: 对每个 anchor 找最难正样本和最难负样本
  5. 支持 hard margin (`max(0, dp - dn + margin)`) 和 soft margin (`log(1 + exp(dp - dn))`)
- 输入：part_based_embeddings `[N, K, D]`, labels `[N]`, parts_visibility `[N, K]` (可选)
- 输出：triplet_loss, trivial_ratio, valid_ratio
- **移植到我们框架的可行性**：**中**
- **额外显存开销估算**：0（纯计算）
- **移植方案**：我们已经有 triplet loss 实现。BPBreID 的部件平均三元组的关键创新是 visibility-aware distance masking。我们的 PAMS 已经有 visibility score，可以借鉴这个思路改进 triplet mining。

### 模块 F: Visibility-Aware Distance Matrix Computation（可见性感知距离计算）
- 文件位置：`torchreid/metrics/distance.py` L87-L247
- 功能：在测试时计算 query-gallery 距离矩阵，只比较双方都可见的部件
- 三种实现：
  1. **无 visibility**：所有部件距离取平均 `[K, Nq, Ng] -> mean -> [Nq, Ng]`
  2. **Binary visibility**：布尔 mask，仅在双方都可见的部件上计算距离。不可见的部件对设为 -1（被排除）。若无共同可见部件，距离设为 max_value
  3. **Continuous visibility**：连续 [0,1] 分数，用 `sqrt(vis_q * vis_g)` 作为距离权重，做加权平均
- 核心公式（continuous）：
  ```
  weight[k] = sqrt(vis_q[k] * vis_g[k])  # 几何平均
  dist = sum(weight[k] * dist_k) / sum(weight[k])
  ```
- **移植到我们框架的可行性**：**高** -- 这正是我们 PAMS 需要的
- **额外显存开销估算**：0（推理时计算）
- **移植方案**：我们的 PAMS 已经有类似的 visibility-aware distance 计算。可以对比验证我们的实现与 BPBreID 的实现是否一致。BPBreID 使用几何平均 `sqrt(vis_q * vis_g)` 作为权重，这比简单的 `min(vis_q, vis_g)` 或 `vis_q * vis_g` 更合理。

### 模块 G: Human Parsing Label Generation (PifPaf -> Part Masks)
- 文件位置：`torchreid/data/masks_transforms/pifpaf_mask_transform.py` L1-L535
- 功能：将 PifPaf 姿态估计模型生成的 36 通道原始 mask（17 个 keypoint heatmaps + 19 个 joint/limb heatmaps）分组合并为各种粒度的部件 mask
- 提供从 1-part 到 14-part 的多种分组方案：
  - **CombinePifPafIntoFourBodyMasks**：head, arms, torso, legs
  - **CombinePifPafIntoFiveBodyMasks**：head, arms, torso, legs, feet
  - **CombinePifPafIntoSixBodyMasks**：head, left_arm, right_arm, torso, left_leg, right_leg
  - **CombinePifPafIntoEightBodyMasks** [常用]：head, left_arm, right_arm, torso, left_leg, right_leg, left_feet, right_feet
  - 等等多种变体
- 合并方式：对同组内所有 keypoint/joint heatmaps 取 max（也支持 sum），然后 clamp 到 [0,1]
- 背景 mask：`1 - max(all_parts_masks)`
- **移植到我们框架的可行性**：**高** -- 这给了我们一个从 pose keypoints 到 part attention masks 的完整流水线
- **额外显存开销估算**：0（数据预处理）
- **移植方案**：
  1. 我们已有 pose keypoints 数据（DWPose 或类似模型离线生成）
  2. 可以直接借鉴 BPBreID 的分组方案将 keypoint heatmaps 合并为 part masks
  3. 重点选择 **FiveBodyMasks**（head, arms, torso, legs, feet）或 **FourBodyMasks**（head, arms, torso, legs）方案，与我们 PAMS 的 5-part 划分对齐
  4. 这些 part masks 可以用作 PixelToPartClassifier 的训练监督信号

### 模块 H: BNClassifier (BatchNorm + Linear 分类器)
- 文件位置：`torchreid/models/bpbreid.py` L398-L425
- 功能：`BN1d(D) -> Linear(D, num_classes)`，先 BN 归一化再做分类。BN 后的特征也作为 inference 时的嵌入（BN features 通常比 raw features 效果更好）
- 关键设计：`bn.bias.requires_grad_(False)` -- BoT trick
- 输入：embeddings `[N, D]`
- 输出：bn_embeddings `[N, D]` (for inference), cls_scores `[N, num_classes]` (for training)
- **移植到我们框架的可行性**：已有类似实现（我们的 PAMS 使用 BN + classifier）

### 模块 I: Multi-level Embedding Extraction (多层级嵌入)
- 文件位置：`bpbreid.py` L116-L259 (forward 方法)
- 功能：从一个 backbone 输出生成 5 类嵌入：
  1. **global**: 全局 GAP `[N, D]`
  2. **foreground**: 前景 attention weighted pooling `[N, D]`
  3. **background**: 背景 attention weighted pooling `[N, D]`
  4. **concat_parts**: K 个部件嵌入拼接 `[N, K*D]`
  5. **parts**: K 个独立部件嵌入 `[N, K, D]`
- 每类嵌入都有独立的 BNClassifier，产生 bn_embeddings 和 cls_scores
- Test time 只使用 `bn_foreground` + `parts`（默认配置）
- **移植到我们框架的可行性**：**中** -- 与 PAMS 的 global + part 结构类似
- **移植方案**：
  1. 我们已有 global embedding + part embeddings
  2. 可以增加 foreground embedding（对所有 part masks 取 max 作为前景 mask，然后做 weighted pooling）
  3. foreground embedding 的好处：它是一个遮挡感知的全局特征（被遮挡区域权重低），比普通 GAP 在遮挡场景下更鲁棒

## 损失函数
1. **GiLt Loss**：`global_id + foreground_id + concat_parts_id + parts_triplet`，各项权重均为 1.0。核心思想是全局用 ID loss，局部用 triplet loss。
2. **Body Part Attention Loss**：像素级 CrossEntropyLoss (label_smoothing=0.1)，权重 0.35。训练 attention mechanism 学习正确的部件分割。
3. **Part-Averaged Triplet Loss**：先合并部件距离再做 batch-hard mining，margin=0.3。支持 visibility-aware 的距离合并。
4. **CrossEntropyLoss (label smooth)**：标准 ID loss with label smoothing。

## 训练 Tricks
- **Backbone**: HRNet-W32 (不做 last stride 修改，输出分辨率自然为 input/4 -> 96x32 for 384x128)
- **Input size**: 384x128 (与我们一致)
- **Data augmentation**: 使用 Albumentations 库，`rc` (random crop) + `re` (random erasing)
- **Batch size**: 64
- **Dim reduction**: after_pooling, 从 backbone 输出 D 降到 512
- **Pooling**: GWAP (Global Weighted Average Pooling) -- 真正的加权平均，优于简单的 GAP 或 GMP
- **Parts num**: 在 Occ-Duke 上使用 8 部件（CombinePifPafIntoEightBodyMasks）
- **Mask filtering**: 训练时不用 visibility filtering（mask_filtering_training=False），测试时启用（mask_filtering_testing=True）
- **Test embeddings**: 使用 `bn_foreground` + `parts`
- **BPA weight**: 0.35 (Body Part Attention Loss 权重)
- **Human parsing labels**: 由 PifPaf + MaskRCNN 离线生成，存储在 `masks/pifpaf_maskrcnn_filtering/` 目录
- **Shared parts classifier**: 默认不共享（每个部件有独立的 BNClassifier），共享时参数量更少但效果略差
- **Binary vs continuous visibility**: 测试时使用 binary visibility（更稳定），训练时使用 continuous visibility

## 对我们框架的改进建议

### 高优先级

1. **引入 Foreground Embedding**：
   - 在我们 PAMS 中，当前只有 global (GAP) + per-part embeddings
   - 可以增加一个 foreground embedding：将所有 part visibility scores 取 max 作为前景 mask，然后做 weighted average pooling
   - 这提供了一个"遮挡感知的全局特征"，在部件级特征不可靠时作为 fallback
   - 实现成本极低，无额外参数，仅需一个额外的 pooling 操作
   - 显存开销：~0

2. **GiLt 损失策略调优**：
   - 当前我们对所有嵌入（global + parts）都用 ID loss + triplet loss
   - BPBreID 的实验表明：**全局特征用 ID loss，部件特征用 triplet loss** 效果最好
   - 原因分析：单个部件（如只有手臂）可能不够判别来做 ID 分类，但足以做距离度量（triplet）
   - 建议实验：移除 per-part ID loss，只保留 per-part triplet loss；同时增强 global/foreground ID loss
   - 显存开销：0（甚至可能减少，因为去掉了 per-part classifier 参数）

3. **Visibility-Aware Triplet Mining 改进**：
   - BPBreID 在 triplet loss 中使用 visibility scores 来 mask 不可见部件的距离
   - 具体做法：`valid_weight = sqrt(vis_anchor * vis_positive/negative)` 用于加权部件距离
   - 这确保 triplet mining 不会被遮挡部件的噪声距离误导
   - 我们的 PAMS 在测试时已有 visibility-aware distance，但训练时的 triplet loss 可能未充分利用

### 中优先级

4. **Learnable Attention vs Fixed Keypoint Attention 实验**：
   - 我们当前用 pose keypoints 硬性生成 part attention（Gaussian heatmaps）
   - 可以实验添加一个 PixelToPartClassifier 学习 soft attention
   - 折中方案：用 keypoint heatmaps 作为 soft supervision（而非 hard assignment）来训练 learnable attention
   - 好处：learnable attention 可以适应 fine-tuning 数据分布，更灵活
   - 风险：需要额外的 parsing labels 或 keypoint heatmaps 作为监督
   - 显存开销：极小（一个 1x1 Conv）

5. **GWAP Pooling 验证**：
   - BPBreID 发现 GWAP 优于 GAP 和 GMP
   - 核心区别：GWAP 将 sum(features * mask) 除以 sum(mask)，而不是固定的空间大小
   - 这对于大小不同的部件（如 torso vs feet）特别重要，确保小部件不会因为空间大小小而被淹没
   - 我们的 PAMS 使用了什么 pooling？需要验证是否已经是 GWAP

6. **Concat Parts Embedding**：
   - BPBreID 除了 per-part embeddings，还有一个 concat_parts embedding (`K*D` 维)
   - 这个嵌入可以用 ID loss 训练，提供一个部件间的联合判别信号
   - 可以作为对 global embedding 的补充

### 低优先级

7. **Test-time Segmentation Refinement**：
   - BPBreID 支持在测试时用 external masks 对 learned attention 做 soft/hard refinement
   - 如果我们有测试集的 pose keypoints，可以在 inference 时进一步精化 part attention

8. **Label Smoothing for Pixel Classification**：
   - BPBreID 的 Body Part Attention Loss 使用 label_smoothing=0.1
   - 这对于 pseudo labels（可能有噪声）很重要

### 与我们 PAMS 的关键差异分析

| 维度 | BPBreID | 我们的 PAMS |
|------|---------|------------|
| Backbone | HRNet-W32 (2048d) | Swin-Tiny (768d) |
| Part attention | Learnable (PixelToPartClassifier) | Fixed (keypoint Gaussian heatmaps) |
| Supervision | PifPaf parsing labels | DWPose keypoints |
| Pooling | GWAP (weighted avg) | 需要验证 |
| Parts num | 8 (on Occ-Duke) | 5 |
| Loss strategy | GiLt: global-ID + part-triplet | ID + triplet for all |
| Visibility | Continuous [0,1] (train off, test on) | Binary/continuous (已有) |
| Test features | bn_foreground + parts | global + parts (concat) |
| Foreground embed | Yes (explicit) | No |
| Background embed | Yes (explicit) | No |
| Dim reduction | after_pooling to 512 | 需要验证 |

### 最有价值的借鉴点（按投入产出排序）
1. **[极低成本] 增加 foreground embedding** -- 0 参数，~1 行代码
2. **[低成本] GiLt 损失策略实验** -- 仅修改 loss 权重配置
3. **[低成本] 验证 GWAP pooling** -- 确认/修改 pooling 实现
4. **[中成本] Visibility-aware triplet mining** -- 修改 triplet loss 实现
5. **[中成本] Learnable attention + BPA loss** -- 需要准备 parsing labels 并添加新模块
