# Paper 8: SOLIDER -- Beyond Appearance: a Semantic Controllable Self-Supervised Learning Framework for Human-Centric Visual Tasks
**来源**: CVPR 2023
**仓库**: https://github.com/tinyvision/SOLIDER
**arXiv 摘要**: SOLIDER uses pseudo semantic labels from unsupervised clustering on LUPerson data, combined with a conditional network and semantic controller, to pretrain a human representation that can be tuned for different downstream tasks by adjusting the ratio of semantic vs. appearance features.

## 代码架构概览
- 核心文件：`swin_transformer.py` (backbone with semantic embedding), `main_solider.py` (training loop), `utils.py` (multi-crop wrapper, mask generation), `demo.py` / `convert_model.py` (downstream usage)
- 模型入口：`SwinTransformer.forward()` in `swin_transformer.py` L1334-L1362
- 预训练流程入口：`train_dino()` in `main_solider.py` L145-L344
- 预训练采用 DINO (teacher-student EMA) 框架 + 语义聚类辅助损失
- Backbone 选择：Swin-Tiny/Small/Base，通过工厂函数 `swin_tiny_patch4_window7_224()` 等创建

### 关键设计
1. **两阶段训练**：先用标准 DINO 自监督预训练，再用 SOLIDER (DINO + Semantic Clustering) 继续训练
2. **Semantic Clustering**：用 KMeans 在 teacher 特征图上聚类生成伪语义标签（前景/背景分离 + K 个部件聚类）
3. **Semantic Controller (conditional network)**：通过 semantic_weight 参数控制特征中语义信息和外观信息的比例

## 可拆解模块清单

### 模块 A: Semantic Embedding (语义控制器)
- 文件位置：`swin_transformer.py` L1217-L1231 (初始化), L1349-L1352 (forward 中应用)
- 功能：在 Swin Transformer 每个 stage 之后，通过一个 per-stage 的 affine transform（由 semantic_weight 向量控制）来调制特征，实现语义/外观特征比例的连续控制
- 输入：`semantic_weight` 二维向量 `[w, 1-w]`，其中 `w` 控制语义比例（w=1 为纯语义，w=0 为纯外观）；stage 输出 token 序列 `[B, L, C]`
- 输出：调制后的 token 序列 `[B, L, C]`（shape 不变）
- 核心公式：`x = x * softplus(sw) + sb`，其中 `sw = Linear(semantic_weight)` 是 per-channel scale，`sb = Linear(semantic_weight)` 是 per-channel bias
- 依赖：无外部依赖，仅需 SOLIDER 预训练权重中包含 `semantic_embed_w` 和 `semantic_embed_b` 参数
- 参数量：每个 stage 有 2 个 Linear(2, C_next)，共 4 个 stage -> 约 `4 * 2 * 2 * C_next` 参数。对于 Swin-Tiny（C=[192, 384, 768, 768]），总计约 `4 * (2*192 + 2*384 + 2*768 + 2*768) * 2 = ~17K` 参数，极轻量
- **移植到我们框架的可行性**：**已移植**。我们的 baseline 已经完整集成了 SOLIDER 的 semantic embedding 机制，通过 `cfg.MODEL.SEMANTIC_WEIGHT` 控制。
- **额外显存开销估算**：~0，参数量极小（<0.1MB），forward 仅做逐元素乘加
- **移植方案**：已在 `model/backbones/swin_transformer.py` 和 `model/backbones/pams.py` 中实现

### 模块 B: Pseudo Semantic Label Generation (KMeans 聚类伪标签)
- 文件位置：`utils.py` L658-L700 (`get_mask` 函数)
- 功能：从 teacher 网络的最后一层特征图中，通过两级 KMeans 聚类生成伪语义标签：(1) 先做前景/背景分离（2-class KMeans on L2 norm），(2) 再在前景区域做 K-class 聚类，按 y 坐标排序对齐部件顺序
- 输入：特征图 `[N, C, H, W]`，部件数 `K`（默认 3）
- 输出：`masks [N_valid, H, W]` 取值 0~K（0=背景），`mask_idxs [N_valid]` 有效样本索引
- 依赖：sklearn KMeans
- **移植到我们框架的可行性**：**低**（用于预训练，非 fine-tuning）
- **额外显存开销估算**：不适用（预训练阶段才用）
- **移植方案**：不需要移植。这是预训练时的伪标签生成，下游 fine-tuning 不使用。我们直接加载 SOLIDER 预训练权重即可获益。

### 模块 C: MultiCropCondWrapper (带条件的多尺度前向)
- 文件位置：`utils.py` L601-L635
- 功能：处理多分辨率输入的 forward pass，每个 crop 附带对应的 semantic_weight 向量
- 输入：多尺度图像列表 `x`，对应的 `semantic_weight` 列表
- 输出：经过 head 投影的输出 + 最后一层特征图
- 依赖：仅用于预训练
- **移植到我们框架的可行性**：**不需要移植**，仅预训练使用

### 模块 D: DINO Loss + Semantic Classification Loss (预训练损失组合)
- 文件位置：`main_solider.py` L356-L426 (train_one_epoch), L474-L528 (DINOLoss)
- 功能：DINO 自蒸馏 loss（teacher-student cross-entropy）+ 语义分类 loss（预测被 mask 遮挡的区域属于哪个语义部件）
- 训练策略关键点：
  - Teacher 用 semantic_weight=1（纯语义）生成特征图来构建聚类标签
  - Teacher 用随机 semantic_weight 做 DINO 蒸馏
  - Student 用同样的随机 semantic_weight 学习匹配 Teacher 输出
  - Student 同时预测被遮挡区域的语义标签
  - loss = DINO_cross_entropy + semantic_weight * part_classification_loss
- **移植到我们框架的可行性**：**不需要**，仅预训练使用

## 损失函数
1. **DINOLoss**：标准 DINO 自蒸馏 cross-entropy loss，teacher 输出做 centering + sharpening，student 做 temperature scaling。`loss = -sum(teacher_softmax * log(student_softmax))`。仅预训练使用。
2. **Semantic Classification Loss**：标准 CrossEntropyLoss (reduction='none')，对 student 的 spatial feature 做 pixel-level 部件分类。通过 semantic_weight 进行加权（semantic_weight 高的样本更关注语义分类 loss）。仅预训练使用。

## 训练 Tricks

### 预训练阶段
- **数据集**：LUPerson（大规模无标签行人数据集）
- **两阶段策略**：先 DINO 100 epoch，再 SOLIDER 10 epoch（从 DINO checkpoint resume）
- **DINO 训练参数**（Swin-Tiny）：
  - Input: 256x128 (global), 128x64 (local crops)
  - 8 local crops + 2 global crops
  - Batch: 48/GPU x 8 GPUs = 384 total
  - AdamW, LR=0.0005 (linear warmup 10 epoch, cosine decay to 1e-6)
  - Weight decay: 0.04 -> 0.4 (cosine schedule)
  - FP16 enabled
  - Gradient clipping: norm 3.0
  - Teacher EMA: 0.996 -> 1.0 (cosine schedule)
- **SOLIDER 续训参数**：
  - 仅 10 epoch（从 DINO checkpoint 继续）
  - LR=0.00005（比 DINO 低 10 倍）
  - partnum=3（聚类为 3 个部件：上/中/下）
  - semantic_loss=1.0
- **Semantic Controller**：
  - 预训练时 semantic_weight 随机采样（randint 生成 0/1）
  - 下游 fine-tuning 时设为固定值（如 ReID 常用 0.2~1.0）
  - softplus 确保 scale 非负

### 下游 Fine-tuning 阶段（SOLIDER-REID）
- **semantic_weight 参数选择**是关键超参数：
  - semantic_weight=1.0：纯语义特征，强调身体部件语义信息
  - semantic_weight=0.0：纯外观特征，强调纹理/颜色等低级特征
  - 中间值（如 0.2）：混合特征，通常 ReID 效果最佳
  - 我们的 baseline 实验表明 semantic_weight=0.2 在 Market-1501 和 Occ-Duke 上效果最好

## semantic_weight 机制深入分析

### 数学原理
在每个 Swin stage `i` 后，特征 `x` 被如下调制：
```
sw_i = Linear_w_i([w, 1-w])  # shape [B, C_i+1]
sb_i = Linear_b_i([w, 1-w])  # shape [B, C_i+1]
x = x * softplus(sw_i) + sb_i
```
其中 `w` 是 `semantic_weight` 标量。

这等价于一个 **Feature-wise Linear Modulation (FiLM)**：
- `softplus(sw)` 控制每个 channel 的 **缩放**（总是正数）
- `sb` 控制每个 channel 的 **偏移**
- `[w, 1-w]` 的二维输入允许网络学习到 w=0 和 w=1 两种极端状态之间的线性插值

### 预训练时的学习
- Teacher 使用 semantic_weight=1 提取纯语义特征来生成聚类伪标签
- Student 使用随机 semantic_weight 来学习——这迫使网络在不同 w 值下都能产生有意义的特征
- 语义分类 loss 被 semantic_weight 加权，w 越大的样本越被鼓励学习语义分类

### 对下游任务的影响
- **ReID**：适度的 semantic_weight（0.2）效果最好，因为 ReID 需要兼顾语义对齐（部件级匹配）和外观判别（纹理/颜色区分身份）
- **Parsing/Pose**：更高的 semantic_weight（接近 1.0）效果更好，因为这些任务本身就是语义任务
- **Detection**：更低的 semantic_weight 效果更好，因为检测更依赖外观特征

### 在我们框架中的使用方式
我们的 `model/backbones/pams.py` 中 `_original_semantic()` 方法：
```python
w = torch.ones(x.shape[0], 1, device=x.device) * self.swin.semantic_weight
semantic_weight = torch.cat([w, 1 - w], dim=-1)
```
在 forward 循环的每个 stage 后：
```python
sw = self.swin.semantic_embed_w[i](semantic_weight).unsqueeze(1)
sb = self.swin.semantic_embed_b[i](semantic_weight).unsqueeze(1)
x = x * F.softplus(sw) + sb
```
这与原始 SOLIDER 完全一致。semantic_weight 通过 `cfg.MODEL.SEMANTIC_WEIGHT` 配置。

## 对我们框架的改进建议

1. **semantic_weight 已被充分利用**：我们的 baseline 已经集成了 SOLIDER 的 semantic embedding 机制，并找到了最佳值 0.2。进一步优化空间有限，但可以尝试：
   - **可学习 semantic_weight**：让 semantic_weight 在 fine-tuning 时作为可学习参数而非固定值（需要注意初始化和学习率）
   - **Per-stage semantic_weight**：不同 stage 使用不同的 semantic_weight 值（浅层更偏外观，深层更偏语义）
   - **Per-part semantic_weight**：对于 PAMS 的不同 part branch，使用不同的 semantic_weight（如 head part 更偏语义，torso part 更偏外观）

2. **利用 SOLIDER 多任务能力**：SOLIDER 预训练权重天然支持 human parsing、pose 等任务。可以尝试在 fine-tuning 时加入辅助的 parsing/pose 预测头，利用预训练中学到的语义特征。这与我们的 PAMS 姿态引导方向完全吻合。

3. **特征图质量**：SOLIDER 在 semantic_weight=1 时产生的特征图具有很强的部件聚类性质（这就是预训练时用来生成伪标签的）。可以尝试在 inference 时用 semantic_weight=1 的特征图来辅助 part attention 的生成，而用 semantic_weight=0.2 的特征做最终的 ID 判别。

4. **Knowledge from pre-training clustering**：预训练时使用 3 个部件的 KMeans 聚类（上/中/下），这与我们 PAMS 的 5-part 划分不同。未来可以尝试用更多的 partnum（如 5）重新做 SOLIDER 预训练，使预训练阶段的语义结构与 fine-tuning 时的 part 划分更对齐。不过这需要重新预训练，成本较高。
