# Paper 12: ISP (Identity-Guided Human Semantic Parsing for Person Re-Identification)
**来源**: ECCV 2020 (Spotlight)
**仓库**: https://github.com/CASIA-IVA-Lab/ISP-reID.git
**摘要**: 提出 Identity-Guided Human Semantic Parsing (ISP) 方法，仅利用行人身份标签（无需额外的 parsing 标注），通过特征聚类自动发现人体部件的像素级语义分割伪标签，再用该伪标签反向指导特征学习，实现了无需额外标注的 part-based ReID。

## 代码架构概览

### 整体结构
```
ISP-reID/
  modeling/
    baseline.py          # 主模型：三分支输出（part/global/foreground）
    backbones/
      cls_hrnet.py       # HRNet32 backbone + ISP 核心逻辑（parsing 分支嵌入在 backbone 中）
  layers/
    __init__.py          # 损失函数组装：make_loss / make_loss_with_center
    parsing_loss.py      # Parsing 损失（像素级 CrossEntropy）
    triplet_loss.py      # Triplet Loss + Label Smooth CE
    center_loss.py       # Center Loss（全局/前景/部件各一个）
  engine/
    trainer.py           # 训练循环，包含周期性聚类更新伪标签的逻辑
    clustering.py        # K-Means 聚类核心实现（前景/背景分离 + 部件聚类）
    inference.py         # 推理评估
    miou.py              # Parsing 伪标签质量评估（mIoU）
  data/
    datasets/
      dataset_loader.py  # 数据集加载，包含训练时 parsing 标签的读取
      occluded_dukemtmcreid.py  # Occluded-Duke 数据集定义
    collate_batch.py     # batch 组装（训练时额外传入 parsing target）
  tools/
    train.py             # 训练入口
    test.py              # 测试入口
    visualize.py         # Parsing 结果可视化
```

### 核心文件
- **模型入口**: `modeling/baseline.py` - `Baseline` 类
- **ISP 核心逻辑**: `modeling/backbones/cls_hrnet.py` - `HighResolutionNet.forward()` (L508-L573)
- **聚类逻辑**: `engine/trainer.py` - `cluster_for_each_identity()` (L181-L238) 和 `adjust_mask_pseudo_labels()` 事件处理器

### 数据流概览
```
输入图片 (3, 256, 128)
    |
    v
HRNet32 四阶段多分辨率特征提取
    |
    v
四尺度特征上采样拼接 -> x (B, 1920, H/4, W/4) --- [bigG=True] --> GAP -> y_g (全局特征)
    |
    v
cls_head: 1x1 Conv -> BN -> ReLU -> x (B, 256, H/4, W/4) --- [bigG=False] --> GAP -> y_g
    |
    v
SpatialAttn -> mask -> x = x * mask (空间注意力加权)
    |
    v
part_cls_layer: 1x1 Conv -> part_cls_score (B, K, H/4, W/4)
    |                                  |
    |                         softmax  |
    v                                  v
part_pred (B, K, H/4, W/4)     parsing_loss(part_cls_score, pseudo_labels)
    |
    |--- 前景: sum(part_pred[:,1:K]) -> GAP(x * foreground_mask) -> y_fore (前景特征)
    |--- 部件: 对每个 part p in [1, K-1]: GAP(x * part_pred[:,p]) -> y_part_p
    |          拼接 -> y_part (部件特征)
    v
三分支 BNNeck + Classifier -> cls_score_part, cls_score_global, cls_score_fore
```

## 核心创新点：Identity-Guided Clustering

ISP 最核心的创新在于 **不需要任何 parsing 标注**，仅依靠 ReID 的身份标签，通过以下循环机制自动生成人体语义分割伪标签：

### 聚类机制详解 (`engine/trainer.py` L181-L238)

1. **按身份分组**: 将训练集按 person ID 分组，每个身份的所有图片一起进行聚类
2. **前景/背景分离** (K=2 聚类):
   - 计算每个像素位置的特征向量 L2 范数 (`np.linalg.norm`)
   - 归一化后，通过 sigmoid 增强 (`1/(1+exp(-5*(2x-1)-3))`) 使前景/背景分离更明显
   - 用 K-Means (K=2) 聚类，范数大的类别为前景
3. **部件聚类** (K=PART_NUM-1 聚类):
   - 只对前景像素的特征向量进行 K-Means 聚类（L2 归一化）
   - **关键**: 聚类后按各类别的平均垂直位置 (`mean_h`) 排序，自上而下分配标签 1, 2, ..., K-1
   - 这保证了从头到脚的语义一致性
4. **保存伪标签**: 将聚类结果保存为 PNG 图片，作为下一轮训练的 parsing 监督信号
5. **周期性更新**: 每 `CLUSTERING.PERIOD` (默认2) 个 epoch 执行一次聚类更新，直到 `CLUSTERING.STOP` (默认101) epoch

### 为什么称为 "Identity-Guided"
因为聚类是 **按身份** 进行的 —— 同一个人的所有图片的特征一起参与聚类。同一个人在不同视角/姿态下，相同语义部件（如头部、上衣）的特征应该相似，因此聚类能自然发现这些部件。这是身份信息隐式指导 parsing 的方式。

## 可拆解模块清单

### 模块 A: 无监督 Part 伪标签生成 (Identity-Guided Clustering)
- **文件位置**: `engine/trainer.py` L149-L238 (`compute_features` + `cluster_for_each_identity`)
- **功能**: 利用 backbone 中间特征的聚类结果，按身份分组生成人体部件的像素级伪标签
- **输入**:
  - `clustering_feat_map`: backbone 输出的特征图 (B, 256, H/4, W/4)，从 `cls_head` 之后、`SpatialAttn` 之前获取
  - 身份标签（用于分组聚类）
- **输出**: 每张图片一个 parsing 伪标签图 (H/4, W/4)，像素值 0=背景, 1~K-1=部件
- **依赖**: faiss 库（GPU K-Means），需要较大 CPU 内存（按身份加载所有特征）
- **移植到我们框架的可行性**: **中**
  - 优点: 完全无需额外标注；不增加推理开销
  - 难点: 我们用 Swin-Tiny 而非 HRNet，需要适配特征图格式；聚类计算量大，需要在每隔 N epoch 的 epoch 开始时运行
  - 但我们已有 ViTPose visibility 信息，可以用更高效的方式替代
- **额外显存开销估算**: 聚类阶段需要额外的前向传播（但是 eval 模式，开销可控约 ~1GB），训练时无额外显存
- **移植方案**:
  1. 在 Swin-Tiny 的最后一层输出 (B, C, H, W) 上执行聚类
  2. 或者直接使用我们已有的 ViTPose 关键点作为 part 定义（更高效，无需聚类）

### 模块 B: 三分支特征提取 (Part + Global + Foreground)
- **文件位置**: `modeling/backbones/cls_hrnet.py` L558-L573, `modeling/baseline.py` L92-L120
- **功能**: 从同一个 backbone 特征图中，通过 parsing prediction 的 soft mask 提取三种互补特征
- **输入**: backbone 特征图 x (B, 256, H, W) + parsing 预测 part_pred (B, K, H, W)
- **输出**:
  - `y_part`: 所有部件特征拼接 (B, 256*(K-1)) — 每个部件一个 256-d 特征向量
  - `y_global`: 全局平均池化特征 (B, 1920) 或 (B, 256)
  - `y_fore`: 前景区域平均特征 (B, 256)
- **核心代码** (`cls_hrnet.py` L562-L569):
  ```python
  # 部件特征: 用 parsing softmax 结果作为 soft attention mask
  for p in range(1, self.part_num):
      y_part.append(self.gap(x * part_pred[:,p,:,:].view(N,1,f_h,f_w)))
  # 前景特征: 所有非背景部件 mask 之和
  y_fore = self.gap(x * torch.sum(part_pred[:,1:self.part_num,:,:], 1).view(N,1,f_h,f_w))
  ```
- **移植到我们框架的可行性**: **高**
  - 这个 soft-mask pooling 的思想非常通用，可以直接用到 Swin-Tiny 的输出上
  - 如果我们用 ViTPose 的关键点热图作为 part 定义，可以直接替换 `part_pred`
- **额外显存开销估算**: ~0.3-0.5GB（主要是额外的 BNNeck 和 Classifier）
- **移植方案**:
  1. 用 ViTPose 热图生成 part mask (17 个关键点 -> 合并为 5-7 个人体区域)
  2. 对 Swin-Tiny 最后的特征图做 soft-mask weighted GAP
  3. 加上各自的 BNNeck + ID Loss + Triplet Loss

### 模块 C: Parsing 分类层 (Part Classification Layer)
- **文件位置**: `modeling/backbones/cls_hrnet.py` L350-L354 (`part_cls_layer`)
- **功能**: 1x1 卷积将特征图映射到 K 类的 parsing prediction score map
- **输入**: x (B, 256, H, W)
- **输出**: part_cls_score (B, K, H, W)，经 softmax 后得到 part_pred
- **核心**: 这是一个非常轻量的分割头（仅 1x1 Conv），通过 parsing_loss 与伪标签对齐
- **移植到我们框架的可行性**: **高**
  - 几乎零额外开销
  - 但需要 parsing 伪标签（来自聚类或外部 parsing 模型）
- **额外显存开销估算**: <0.1GB
- **移植方案**: 在 Swin-Tiny 最后特征图上加一个 1x1 Conv(C_in, K)，用关键点衍生的 part mask 监督

### 模块 D: 空间注意力 (SpatialAttn)
- **文件位置**: `modeling/backbones/cls_hrnet.py` L258-L271
- **功能**: 简单的空间注意力机制，抑制背景区域
- **输入**: x (B, 256, H, W)
- **输出**: attention mask (B, 1, H, W)，值域 [0,1]
- **核心代码**:
  ```python
  x = F.relu(self.conv1(x))        # (B, 256, H, W) -> (B, 1, H/2, W/2)
  x = F.upsample(x, scale_factor=2) # (B, 1, H, W)
  x = self.conv2(x)                 # (B, 1, H, W)
  x = torch.sigmoid(x)
  ```
- **移植到我们框架的可行性**: **高**
  - 非常轻量，只有 2 个 Conv 层
  - 但论文作者自己注释说 "can be removed"，效果有限
- **额外显存开销估算**: <0.05GB
- **移植方案**: 可选加入，作为 Swin-Tiny 输出后的简单前景增强

### 模块 E: Aligned Re-ID Matching (ARM)
- **文件位置**: `data/datasets/eval_reid.py` L67-L118 (`arm_eval_func`), `utils/reid_metric.py` L12-L67 (`R1_mAP_arm`)
- **功能**: 测试时利用 parsing 可见性信息进行对齐匹配 —— 只计算 query 和 gallery 共同可见部件的相似度
- **输入**:
  - query/gallery 的全局+前景特征 (g_f_feat)
  - query/gallery 的部件特征 (part_feat) (B, K-1, 256)
  - query/gallery 的部件可见性 (part_visible) (B, K-1) 二值向量
- **输出**: 距离矩阵 / 排序结果
- **核心逻辑** (`eval_reid.py` L87-L101):
  ```python
  overlap = gpl * qpl         # 共同可见的部件掩码
  s = q * g                    # 逐部件余弦相似度
  s = s.sum(2)                 # 每个部件的相似度
  s = (s+1)/2                  # 映射到 [0,1]
  s = s * overlap              # 只保留共同可见部件的相似度
  s = (s.sum(1) + s2) / (overlap.sum(1) + 1)  # 加权平均（加上全局特征相似度）
  ```
- **移植到我们框架的可行性**: **非常高，且与我们的 ViTPose visibility 完美对应**
  - ViTPose 的 visibility 向量直接给出了每个关键点的可见性
  - 可以将关键点分组为人体区域，生成 part-level visibility
  - 测试时只匹配共同可见区域
- **额外显存开销估算**: 仅推理时计算，无训练开销
- **移植方案**:
  1. 训练时用 ViTPose visibility 指导 part 特征学习
  2. 推理时用 visibility 向量构建 overlap 掩码
  3. 计算 query-gallery 的 visibility-aware 距离

### 模块 F: 联合 Random Erasing + Parsing 标签同步
- **文件位置**: `data/datasets/dataset_loader.py` L69-L114 (`RandomErasing`), L116-L139 (`HorizontalFlip`)
- **功能**: 数据增强时，同步修改图像和 parsing 标签
  - Random Erasing: 擦除区域的 parsing 标签设为 0 (背景)
  - Horizontal Flip: 同步翻转 parsing 标签
- **移植到我们框架的可行性**: **高**
  - 如果我们使用 part mask 监督，需要在数据增强时同步处理
- **额外显存开销估算**: 无

## 损失函数

ISP 的总损失由 7 个（或 10 个，含 center loss）子项组成 (`layers/__init__.py` L28-L37):

```python
L_total = L_ce(global) + L_ce(foreground) + L_ce(part)
        + lambda_p * L_parsing(part_cls_score, pseudo_labels)
        + L_triplet(global) + L_triplet(foreground) + L_triplet(part)
        [+ L_center(global) + L_center(foreground) + L_center(part)]
```

| 损失 | 公式/核心思想 | 权重 | 可否直接用 |
|------|---------------|------|-----------|
| Label Smooth CE (x3) | 对 global/foreground/part 三个分支各一个 | 1.0 | 可以 |
| Parsing CE | 像素级交叉熵，监督 part_cls_layer 输出与伪标签对齐 | 0.1 (PARSING_LOSS_WEIGHT) | 需要 parsing 标签 |
| Hard Triplet (x3) | margin=0.3，对三个分支各一个 | 1.0 | 可以 |
| Center Loss (x3) | 拉近同类特征到类中心，对三个分支各一个 | 0.0005 (CENTER_LOSS_WEIGHT) | 可以（需额外优化器） |

**关键观察**:
- Parsing Loss 权重很小 (0.1)，说明 parsing 监督是辅助信号，主要还是靠 ReID loss
- 三个分支（global/foreground/part）各自有独立的 CE + Triplet + Center loss，是完全对称的设计
- Center Loss 需要单独的优化器（学习率 0.5），梯度需要缩放 (1/center_loss_weight)

## 训练 Tricks

### 关键超参数
- **Backbone**: HRNet32（通道数 32/64/128/256）
- **输入尺寸**: 256x128 (H x W)
- **Batch Size**: 64（softmax_triplet sampler，每个身份 4 张图）
- **优化器**: Adam, LR=3.5e-4, Weight Decay=5e-4
- **LR 调度**: Warmup 10 iter (factor=0.01) + MultiStep (40, 70) + Gamma=0.1
- **总 Epochs**: 120
- **Part 数量**: 6 或 7（含背景类，所以实际部件数为 5 或 6）

### 聚类相关
- **聚类算法**: K-Means（使用 faiss GPU 版本）
- **聚类周期**: 每 2 个 epoch 更新一次伪标签（`CLUSTERING.PERIOD=2`）
- **聚类停止**: 第 101 个 epoch 后停止聚类（`CLUSTERING.STOP=101`）
- **前景增强**: sigmoid 增强 `1/(1+exp(-5*(2x-1)-3))`（`CLUSTERING.ENHANCED=True`，但 Occluded-Duke 上设为 False）
- **初始化**: 如果 parsing 标签文件不存在，使用均匀水平条纹初始化（按高度等分为 5/6 个条纹）

### 数据增强
- Random Horizontal Flip (p=0.5)
- Random Erasing (p=0.5)，同步修改 parsing 标签
- Padding=10 + Random Crop

### 推理策略
- **无 ARM**: 拼接 (part + global + foreground) 特征做全局距离计算
- **有 ARM**: 用 parsing 可见性做对齐匹配，只计算共同可见部件的相似度
  - 可见性判断: `argmax(softmax(part_cls_score))` 在空间维度上，检查每个 part 类别是否出现

### 特殊设计
- **bigG 模式**: 全局特征使用四尺度拼接后的 1920-d 特征（而非 cls_head 之后的 256-d），增强全局表达力
- **聚类按身份分组**: 每个身份的所有图片一起聚类，利用身份一致性约束
- **部件排序**: 聚类后按平均垂直位置排序标签，保证从头到脚的语义一致性

## 该工作的局限性 / 未解决的问题

### 1. 伪标签质量受限于聚类
- 聚类结果依赖于 backbone 特征的质量，早期训练时特征不成熟，伪标签噪声大
- K-Means 只能产生凸形状的区域分割，无法准确建模复杂的人体部件边界
- 部件数 K 是手动设定的超参数，不同人体状态（遮挡、姿态变化）可能需要不同的 K

### 2. 聚类效率问题
- 每隔 2 个 epoch 需要对整个训练集做前向传播 + 按身份分组聚类
- 随着训练集规模增大，聚类时间显著增加
- 依赖 faiss GPU 版本，增加了部署复杂度

### 3. 无法处理严重遮挡
- 当人被严重遮挡时，前景/背景聚类可能失败（遮挡物可能被误认为前景）
- 部件聚类假设同一身份不同视角下部件一致，但遮挡会破坏这个假设
- **ISP 没有显式的遮挡检测机制**，ARM 只是一个事后补救

### 4. Backbone 局限
- ISP 使用 HRNet32 作为 backbone，这是一个 CNN 架构
- 在 Transformer 架构（如 Swin-Tiny）上，特征图的语义与 CNN 不同，聚类策略可能需要调整
- HRNet 的多分辨率特征对于 parsing 有天然优势，Swin-Tiny 的窗口注意力可能不如 HRNet 适合像素级任务

### 5. 推理复杂度
- ARM 推理需要逐 query 逐 gallery 计算对齐距离，复杂度为 O(N_q * N_g * K)
- 在大规模检索场景下可能成为瓶颈

### 6. 身份引导聚类的假设过强
- 假设同一身份的人在不同图片中穿着相同，但 ReID 数据集中同一人可能换衣服
- 假设同一身份的部件特征应该聚在一起，但在极端姿态差异下未必成立

## 对我们框架的改进建议

### 建议 1: 用 ViTPose Visibility 替代 ISP 的聚类 Parsing（最推荐）
ISP 最大的创新点（identity-guided clustering）也是其最大的痛点（慢、不准确、依赖 faiss）。我们已经有 ViTPose 提供的关键点坐标 + visibility 信息，可以：

1. **将 17 个关键点分组为 5-7 个人体区域**（头部、躯干、左臂、右臂、左腿、右腿）
2. **用关键点坐标生成 Gaussian 热图作为 part mask**（类似 PFD）
3. **用 visibility 向量直接判断部件可见性**，无需通过聚类推断

这样我们获得了 ISP 的全部好处（part-based 特征 + 可见性感知匹配），同时避开了聚类的所有问题。

### 建议 2: 借鉴 ARM 的对齐匹配策略
ARM 的核心思想（只匹配共同可见部件）与我们的 ViTPose visibility 方案完美匹配：
- 训练时: 用 visibility 对部件 triplet loss 进行掩码（不可见部件不参与 triplet 计算）
- 推理时: 用 visibility 构建 overlap mask，加权计算部件相似度

**具体实现**:
```python
# 测试时对齐匹配
overlap = (q_vis > 0.5) & (g_vis > 0.5)  # 共同可见区域
part_sim = cosine_sim(q_part_feat, g_part_feat)  # (K,)
aligned_sim = (part_sim * overlap).sum() / (overlap.sum() + 1)
total_sim = alpha * global_sim + (1-alpha) * aligned_sim
```

### 建议 3: Soft-Mask Pooling 替代 Hard Partition
ISP 用 softmax 后的 parsing prediction 作为 soft attention mask 来提取部件特征，这比硬分割（PCB 风格的水平切条）更灵活。我们可以：

1. 用 ViTPose 热图生成 soft part mask
2. 对 Swin-Tiny 输出的特征图做 soft-mask weighted GAP
3. 每个部件生成一个独立的 embedding

### 建议 4: 前景感知作为 baseline 增强
ISP 的前景/背景分离（即使不做部件分割）本身就很有价值：
- 用 ViTPose 关键点的整体置信度作为前景先验
- 对 Swin-Tiny 特征图做前景加权池化
- 这是一个零开销的增强策略

### 建议 5: Parsing Loss 作为辅助损失（轻量化版本）
不需要完整的 ISP 聚类机制，可以用 ViTPose 热图作为监督信号：
- 在 Swin-Tiny 最后一层加一个 1x1 Conv 做 part classification
- 用关键点衍生的 part mask 作为 "伪 ground truth"
- 权重设为 0.1（与 ISP 一致），作为正则化辅助损失
- 目的: 强制 backbone 学习 part-aware 的特征表示

### 与 visibility-guided approach 的关联

ISP 的 parsing 可见性判断（`argmax(softmax(part_cls_score))` 在空间维度检查每个 part 是否出现）与我们 ViTPose 的 visibility 向量在功能上完全等价，但有以下差异：

| 维度 | ISP Parsing Visibility | ViTPose Visibility |
|------|----------------------|-------------------|
| 来源 | 模型自身的 parsing 预测 | 外部姿态模型预测 |
| 粒度 | K-1 个人体区域 (粗粒度) | 17 个关键点 (细粒度) |
| 准确性 | 取决于聚类质量，早期不准 | 预训练姿态模型提供，始终可靠 |
| 额外开销 | 需要聚类 + parsing head | 预处理一次，后续零开销 |
| 语义 | 反映"模型能看到哪些区域" | 反映"哪些关键点实际可见（未被遮挡）" |

**关键洞察**: ISP 的 parsing visibility 是一个 "模型内部估计"，而我们的 ViTPose visibility 是一个 "外部先验"。两者可以互补：
- 训练时用 ViTPose visibility 作为可靠监督信号
- 模型同时学习自己的 part-level 可见性估计（类似 ISP）
- 推理时不依赖 ViTPose，用模型自己估计的可见性（如果训练成功的话），或仍用 ViTPose（更稳健）

这一思路可以发展为一个创新点: **"用外部姿态先验引导模型学习遮挡感知的特征，使其在推理时不再依赖外部姿态模型"** —— 这本质上是一种知识蒸馏思想，从姿态模型向 ReID 模型蒸馏遮挡/可见性知识。
