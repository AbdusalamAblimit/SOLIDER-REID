# Paper 6: TransReID -- Transformer-based Object Re-Identification
**来源**: ICCV 2021
**仓库**: https://github.com/damo-cv/TransReID
**arXiv 摘要**: 首次系统性地将纯 Vision Transformer 应用于 ReID 任务，提出 SIE（Side Information Embedding）和 JPM（Jigsaw Patch Module）两个关键模块，在多个 person/vehicle ReID 数据集上达到 SOTA。

## 代码架构概览
- 核心文件：`model/make_model.py`（模型入口，定义 `build_transformer` 和 `build_transformer_local`）
- Backbone 定义：`model/backbones/vit_pytorch.py`（ViT 基础架构 + SIE 嵌入 + 重叠 patch embedding）
- 损失函数：`loss/make_loss.py`（ID loss + triplet loss 的组合），支持 label smoothing、arcface/cosface/circle loss 等变体
- 配置系统：`config/defaults.py`（YACS 配置），`configs/` 下按数据集分子目录

模型入口函数 `make_model()` 根据 `cfg.MODEL.JPM` 决定构建：
- `build_transformer`：纯全局特征分支（baseline）
- `build_transformer_local`：全局 + JPM 局部特征分支（完整 TransReID）

## 可拆解模块清单

### 模块 A: Side Information Embedding (SIE)
- 文件位置：`model/backbones/vit_pytorch.py` L291-L331（`TransReID.__init__`），L375-L389（`forward_features`）
- 功能：将 camera ID 和/或 viewpoint ID 编码为可学习的 embedding，加到 position embedding 上，显式注入拍摄视角信息以帮助模型区分跨相机外观差异。
- 输入：`camera_id` (int), `view_id` (int)，由 dataloader 提供
- 输出：直接加到 `pos_embed` 上，不改变 token 维度
- 依赖：需要数据集提供 camera label（Occluded-Duke 有 camera label，无 view label）
- 实现细节：
  - 仅 camera: `sie_embed = nn.Parameter(torch.zeros(camera_num, 1, embed_dim))`，以 truncated normal (std=0.02) 初始化
  - 前向传播：`x = x + pos_embed + sie_xishu * sie_embed[camera_id]`
  - `sie_xishu`（系数）默认为 3.0，控制 SIE 的强度
- **移植到我们框架的可行性**：高
  - 我们的 SOLIDER-REID 已经支持 `SIE_CAMERA` 和 `SIE_COE` 配置项
  - Swin-Tiny 有 4 个 stage，每个 stage 的 patch embedding 后都可以加 SIE
  - 但是 Swin 的 patch embedding 机制与 ViT 不同（层级式 vs 扁平式），需要在每个 stage 的输入处注入
- **额外显存开销估算**：几乎为 0（只增加一个 `[C, 1, D]` 的参数矩阵，C 为相机数通常 < 10）
- **移植方案**：
  1. 在 Swin-Tiny 的 patch merging 之前或 stage 3/4 的输入处加入 camera embedding
  2. 由于 Swin 的 token 数随 stage 递减（96x32 -> 48x16 -> 24x8 -> 12x4），SIE 应作为全局偏移（broadcast 到所有 token），而非逐 token 不同
  3. 简化方案：只在最终输出的全局特征上加 camera embedding（类似 TransReID 加到 cls_token 的效果）
  4. 注意：我们框架已经原生支持此功能，只需确认 config 中 `SIE_CAMERA: True` 即可

### 模块 B: Jigsaw Patch Module (JPM)
- 文件位置：`model/make_model.py` L8-L25（`shuffle_unit` 函数），L215-L371（`build_transformer_local` 类）
- 功能：将 backbone 输出的 patch tokens 打乱重组为多个局部组，每组通过共享的额外 Transformer block 提取局部特征，产生多个互补的 part-level 表征。核心思想是通过 shift + group shuffle 打破空间连续性，迫使每个局部分支学到不同的部件信息。
- 输入：`features` [B, N+1, D]（backbone 倒数第二层的输出，包含 cls_token）
- 输出：`[global_feat, local_feat_1, ..., local_feat_4]`，每个 [B, D]
- 依赖：无外部依赖
- 实现细节：
  1. **Shift 操作**：将 patch tokens 循环移位 `shift_num`（默认 5）个位置，打乱原始空间顺序
  2. **Group Shuffle**：将 shifted tokens reshape 为 `[B, group, N/group, D]`，转置 dim1 和 dim2，再 flatten 回来。group 默认为 2
  3. **Divide**：将 shuffled tokens 等分为 `divide_length`（默认 4）个组
  4. **Local Branch**：每组 tokens 与 cls_token 拼接后送入共享的 `b2` 分支（deepcopy 的最后一个 Transformer block + LayerNorm），取 cls_token 输出作为该组的局部特征
  5. **Global Branch**：原始 features 经过 `b1` 分支（另一个 deepcopy），取 cls_token 作为全局特征
  6. **BNNeck**：每个分支都有独立的 BN + classifier
  7. **推理拼接**：全局 + 4 个局部（局部除以 4 做 scale），cat 成最终特征
  8. **Loss**：全局和每个局部分支都有独立的 ID loss + triplet loss，以 0.5:0.5 的权重组合（全局占 50%，局部平均占 50%）
- **移植到我们框架的可行性**：中
  - JPM 的核心设计是针对 ViT 的扁平 patch token 序列的，ViT 输出 [B, 128+1, 768] 的 tokens
  - 我们的 Swin-Tiny 输出的是层级特征图（最终 stage 为 [B, 12x4, 768]），已经是 48 个 token
  - 我们已有 PAMS 模块做 pose-aware part features，理念上与 JPM 有重叠
  - JPM 不用姿态信息做分组，而是用 random shuffle，两种思路互补
- **额外显存开销估算**：约 0.5-0.8G
  - 需要 deepcopy 2 份 Transformer block（Swin-Tiny 最后一个 block 约 13M 参数）
  - 但每个 local branch 只处理 1/4 的 token，计算量并非线性增加
  - 4 个 local branch 共享 b2 权重，总额外参数约 26M（2 个 block copy），FP16 下约 0.05G 参数 + 前向激活约 0.4-0.7G
- **移植方案**：
  1. 在 Swin-Tiny 最后一个 stage 的输出上操作（12x4=48 tokens, 768-dim）
  2. 将 48 个 token shift + shuffle 后分成 4 组（每组 12 个 token）
  3. 每组经过共享的额外 Swin block（或简化为 Linear + Attention）提取 local feature
  4. 但注意：Swin 使用 window attention，直接 deepcopy block 后喂入打乱顺序的 token 可能破坏 window 结构。需要改为使用全局 attention 的 block（如普通 Transformer block），或改用 adaptive pooling
  5. **简化替代方案**：不用 shuffle，直接在 Swin 输出的 12x4 特征图上做水平 4-strip 切分（每条 3x4=12 tokens），这更适合 person ReID 的竖直身体布局，且与 Swin 的 window 结构兼容。实际上这就退化为标准的 horizontal part pooling，我们的 PAMS 方案更优

### 模块 C: Overlapping Patch Embedding (OLP)
- 文件位置：`model/backbones/vit_pytorch.py` L251-L288（`PatchEmbed_overlap` 类）
- 功能：使用 stride < patch_size 的卷积进行 patch embedding，产生重叠的 patch tokens，增加空间分辨率
- 输入：`[B, 3, H, W]` 图像
- 输出：`[B, num_patches, embed_dim]`，num_patches 因 stride 更小而增多
- 实现细节：
  - 标准 ViT: stride=16, patch_size=16 -> 对 [256,128] 产生 16x8=128 tokens
  - OLP: stride=12, patch_size=16 -> 对 [256,128] 产生 21x10=210 tokens（增加 64%）
  - `num_y = (H - P) // S + 1`, `num_x = (W - P) // S + 1`
  - 位置编码通过双线性插值从 [14x14] 调整到 [num_y x num_x]
- **移植到我们框架的可行性**：低
  - Swin-Tiny 使用 4x4 patch + 层级 window attention，patch embedding 机制完全不同
  - Swin 的空间分辨率已经通过层级设计保持较高（不像 ViT 一步到 16x下采样）
  - 不适用于 Swin 架构
- **额外显存开销估算**：N/A（不适用）
- **移植方案**：不推荐移植

### 模块 D: BNNeck (Batch Normalization Neck)
- 文件位置：`model/make_model.py` L178-L181, L284-L298
- 功能：在特征输出后加一层 BN，BN 后的特征用于 ID loss（分类），BN 前的原始特征用于 triplet loss（度量学习）。这是 ReID 的标准做法（源自 BoT baseline）
- 实现细节：
  - `bottleneck = nn.BatchNorm1d(in_planes)`
  - `bottleneck.bias.requires_grad_(False)` -- bias 固定为 0
  - Kaiming 初始化：weight=1, bias=0
  - 训练时：ID loss 用 BN 后特征，triplet loss 用 BN 前特征
  - 测试时：`NECK_FEAT='before'` 用原始特征（TransReID 推荐），`'after'` 用 BN 后特征
- **移植到我们框架的可行性**：已有
  - 我们的 SOLIDER-REID 已经实现了 BNNeck
- **移植方案**：无需移植，已是 baseline 的一部分

## 损失函数

### ID Loss（Cross Entropy with Label Smoothing）
- `loss/softmax_loss.py`: `CrossEntropyLabelSmooth(num_classes, epsilon=0.1)`
- 公式：`y_smooth = (1 - eps) * y_onehot + eps / K`，然后计算 `-(y_smooth * log_softmax(pred)).mean(0).sum()`
- 在 JPM 模式下：全局分支 ID loss 权重 0.5，4 个局部分支平均后权重 0.5
- 可否直接用：已有类似实现

### Triplet Loss（Hard Mining, Soft Margin）
- `loss/triplet_loss.py`: `TripletLoss(margin=None)` 使用 SoftMarginLoss（无固定 margin）
- 欧氏距离 hard mining: 对每个 anchor，找最远正样本和最近负样本
- `hard_factor=0.0`（默认不使用 hard factor 缩放）
- JPM 模式下同样 0.5:0.5 权重
- 可否直接用：我们已有 triplet loss

### 支持的 metric learning losses
- Arcface, Cosface, AMSoftmax, CircleLoss（`loss/metric_learning.py`）
- 通过 `MODEL.ID_LOSS_TYPE` 配置切换
- TransReID 默认使用 softmax（普通 CE），配合 soft triplet
- 可否直接用：我们已集成

## 训练 Tricks

### 超参数（OCC-Duke 配置 vit_transreid.yml）
- Backbone: ViT-Base (768-dim, 12 heads, 12 layers)
- Input: [256, 128]
- Optimizer: SGD, base_lr=0.008
- Epochs: 120
- Batch size: 64 (16 IDs x 4 instances)
- Warmup: linear, 5 epochs
- LR schedule: cosine annealing（通过 config 中未显式列出的 warmup + steps）
- Weight decay: 1e-4
- Label smoothing: off（OCC-Duke 配置）
- Triplet: soft margin (no fixed margin)
- SIE: camera only, coefficient=3.0
- JPM: shift_num=5, shuffle_groups=2, divide_length=4, re_arrange=True

### 数据增强
- Random horizontal flip: p=0.5
- Random padding (10px) + random crop
- Random erasing: p=0.5, pixel mode
- Pixel mean/std: [0.5, 0.5, 0.5]（注意不是 ImageNet 标准值）

### 关键设计决策
1. **SIE 系数 3.0**：比 1.0 大很多，说明 camera embedding 需要较强的幅度才能产生效果
2. **JPM 的 shift_num=5**：不是简单地按空间顺序切分，而是先打乱再分组，增加多样性
3. **局部分支共享权重**：4 个局部组共享 b2 block，而非各自独立，节省参数
4. **全局分支独立**：b1 block 独立于 b2，全局和局部特征由不同的 head 处理
5. **推理时局部特征 /4 缩放**：避免局部特征主导距离计算

### 报告性能（OCC-Duke, Size 256, ViT-Base）
| 方法 | mAP | Rank-1 |
|------|-----|--------|
| Baseline (ViT) | 53.8 | 61.1 |
| TransReID (ViT, SIE+JPM) | 59.5 | 67.4 |
| 提升 | +5.7 | +6.3 |

## 对我们框架的改进建议

1. **SIE Camera Embedding（优先级：高，可直接启用）**
   - 我们的框架已经支持 `SIE_CAMERA`，但当前使用 Swin-Tiny 作为 backbone 时可能未充分利用
   - 建议：确认在 Occluded-Duke 上启用 camera embedding（Swin 版本），观察是否有提升
   - 注意 SIE 系数调优：TransReID 用 3.0，我们的框架默认值可能不同

2. **JPM 思路的简化适配（优先级：中）**
   - 核心思想：对 backbone 输出做多种不同的局部分组 pooling，增加特征多样性
   - 我们已经有 PAMS 做 pose-guided part features，可以考虑额外增加一个 "shuffle-based" 分组作为互补
   - 但 PAMS 的 pose-aware 分组已经比 random shuffle 更有语义意义，JPM 的增益可能有限
   - 简化方案：不做 shuffle，直接在 PAMS 的 part features 之外增加一个 horizontal strip pooling 分支

3. **Loss 配置参考（优先级：中）**
   - TransReID 在 OCC-Duke 上用 soft triplet (no margin) + no label smoothing，效果好
   - 我们当前的 loss 配置可以参考这一设置
   - JPM 的 loss 权重策略（全局:局部 = 0.5:0.5）也值得参考

4. **数据增强参考（优先级：低）**
   - TransReID 使用 [0.5, 0.5, 0.5] 的 pixel mean/std（ViT 预训练标准）
   - 我们使用 SOLIDER 预训练的 Swin-Tiny，应该使用 SOLIDER 预训练时对应的 mean/std
   - Random erasing probability 0.5 是标准配置，可以保持

5. **不推荐移植的部分**
   - OLP (Overlapping Patch Embedding)：不适用于 Swin 层级架构
   - b1/b2 额外 block：需要处理 Swin window attention 的兼容性问题，收益不确定
   - 全套 JPM 移植到 Swin：重新实现成本高，且我们的 PAMS 已覆盖类似功能
