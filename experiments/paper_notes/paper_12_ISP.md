# Paper 12: ISP (Identity-Guided Human Semantic Parsing for Person Re-Identification)
**来源**: ECCV 2020 (Spotlight)
**仓库**: https://github.com/CASIA-IVA-Lab/ISP-reID
**arXiv 摘要**: 提出 ISP 方法，仅使用行人 ID 标签（无需额外的人体解析标注），通过聚类和迭代优化自动生成像素级的人体部件伪标签，实现联合语义分割与 ReID 训练，在 Occluded-Duke 上取得了当时的 SOTA。

## 代码架构概览
- Backbone：HRNet-W32（多分辨率特征融合网络），256-dim 最终特征
- 核心文件：`modeling/backbones/cls_hrnet.py` — HRNet backbone + 部件分割头 + 空间注意力
- 模型入口：`modeling/baseline.py` — `Baseline` 类，封装 backbone + BNNeck + 分类器
- 训练逻辑：`engine/trainer.py` — 包含聚类更新伪标签的完整训练流程
- 聚类模块：`engine/clustering.py` — K-means 聚类 + PIC 聚类
- 损失函数：`layers/__init__.py` — 组合 CE + Triplet + Center + Parsing Loss
- 评估：`utils/reid_metric.py` — 支持 ARM（Aligned ReID Matching）的可见性感知评估
- 配置：`config/defaults.py` + `configs/softmax_triplet.yml`

### 文件结构
```
modeling/
├── baseline.py                       # 核心：Baseline 模型（3分支：part/global/fore）
├── backbones/
│   └── cls_hrnet.py                  # HRNet + 部件分割头 + SpatialAttn
engine/
├── trainer.py                        # 训练主循环，含聚类更新
├── clustering.py                     # K-means / PIC 聚类实现
├── inference.py                      # 推理
└── miou.py                           # mIoU 评估（评估伪标签质量）
layers/
├── __init__.py                       # make_loss / make_loss_with_center
├── triplet_loss.py
├── center_loss.py
├── parsing_loss.py                   # 像素级交叉熵（部件分割损失）
├── cluster_loss.py                   # ClusterLoss / ClusterLoss_local
└── range_loss.py
data/
├── datasets/
│   ├── occluded_dukemtmcreid.py      # Occluded-Duke 数据集
│   └── dataset_loader.py             # 含 parsing label 加载的 DataLoader
├── collate_batch.py                  # 自定义 batch collate（含 parsing target）
└── transforms/                       # 数据增强
utils/
├── reid_metric.py                    # R1_mAP + R1_mAP_arm（可见性感知评估）
└── re_ranking.py                     # Re-ranking
config/
└── defaults.py                       # 完整配置定义
configs/
└── softmax_triplet.yml               # 训练配置
```

## 可拆解模块清单

### 模块 A: Identity-Guided Clustering（身份引导的聚类伪标签生成）
- 文件位置：`engine/trainer.py` L149-L238（`compute_features` + `cluster_for_each_identity`）和 `engine/clustering.py` 全文
- 功能：ISP 的核心创新。不需要人体解析标注，仅用 ID 标签就能生成像素级的部件伪标签：
  1. **特征提取**：用当前模型提取所有训练图片的特征图 [B, C, H, W]
  2. **前景/背景聚类**：计算每个像素的 L2 范数，归一化后用 sigmoid 增强（将前景/背景的差异放大），然后对所有像素做 K=2 的 K-means，分为前景/背景
  3. **身份内部件聚类**：对每个 ID，取其所有图片的前景像素特征，做 K=part_num-1 的 K-means 聚类，得到部件标签
  4. **部件标签排序**：按聚类中心的 y 坐标排序（从上到下），保证部件标签的语义一致性（head=1, torso=2, legs=3, ...）
  5. **保存伪标签**：生成与原图同分辨率的 parsing map，保存为 PNG
  6. **周期性更新**：每 2 个 epoch 重新聚类一次，直到 epoch 101（CLUSTERING.STOP）
- 输入：模型当前参数 + 训练集图片
- 输出：像素级伪标签图 [H, W]，每个像素值为 0(bg)/1(part1)/2(part2)/.../K(partK)
- 依赖：faiss 库（GPU 加速的 K-means）
- **移植到我们框架的可行性**：中
- **额外显存开销估算**：聚类过程在训练间隙执行，不增加训练时显存。但需要额外的前向传播时间（每2个epoch约数分钟）
- **移植方案**：
  - 方案1（推荐）：在训练前离线用我们的 Swin-Tiny 提取特征图，做一次聚类生成伪标签，然后作为固定标签训练。这避免了在线聚类的复杂性。
  - 方案2：完整移植在线聚类，每 N 个 epoch 更新伪标签。需要安装 faiss-gpu。
  - 关键挑战：Swin-Tiny 的特征图分辨率为 12x4（384/32, 128/32），比 HRNet 的 96x32 低很多，聚类效果可能打折。需要用 Swin 的中间 stage（如 stage3 的 24x8）来提高分辨率。

### 模块 B: Parsing Loss（部件分割损失）
- 文件位置：`layers/parsing_loss.py` 全文
- 功能：标准的像素级交叉熵损失，用于监督部件分割头的输出：
  ```python
  loss = CrossEntropyLoss(part_pred_score, parsing_target)
  # part_pred_score: [B, K, H, W]（K 类部件预测）
  # parsing_target: [B, H, W]（伪标签）
  ```
  - 如果预测和标签分辨率不同，自动双线性插值对齐
  - 损失权重：0.1（SOLVER.PARSING_LOSS_WEIGHT）
- 输入：part_pred_score [B, K, H, W]（分割预测），target [B, H, W]（像素标签）
- 输出：scalar loss
- 依赖：需要像素级伪标签
- **移植到我们框架的可行性**：高
- **额外显存开销估算**：<0.1G（1x1 卷积分割头）
- **移植方案**：在 Swin-Tiny 最后一层特征上加一个 1x1 卷积分割头（768 -> K 类），用关键点位置生成的伪标签做监督。伪标签可以很简单：根据 17 个关键点位置，将 part assignment map 下采样到特征图分辨率（12x4），作为 parsing target。这比 ISP 的 K-means 聚类更直接，因为我们已有关键点信息。

### 模块 C: 三分支特征架构（Part / Global / Foreground）
- 文件位置：`modeling/baseline.py` L34-L128
- 功能：模型输出三组特征，各有独立的 BNNeck + 分类器：
  1. **Part 分支**：concat 所有部件的 GAP 特征 → [B, 256*(K-1)]（K-1 个前景部件）
  2. **Global 分支**：全图特征的 GAP → [B, 1920]（HRNet 多尺度融合特征）或 [B, 256]
  3. **Foreground 分支**：前景区域的加权 GAP → [B, 256]（用 softmax 过的 part_pred 中前景部分加权）
  - 训练时：三组特征各自算 CE + Triplet + Center loss
  - 测试时：
    - 无 ARM：三组特征 concat → [B, 256*(K-1)+1920+256]
    - 有 ARM：使用 part 可见性做对齐匹配
- 输入：原始图片 [B, 3, 384, 128]
- 输出：
  - 训练：cls_score_part, cls_score_global, cls_score_fore, y_part, y_global, y_fore, part_pd_score
  - 测试（ARM）：g_f_feat [B, 1920+256], part_feat [B, K-1, 256], visible_part [B, K-1]
- 依赖：HRNet backbone + 部件分割头
- **移植到我们框架的可行性**：高（概念已部分在我们 PAMS 中实现）
- **额外显存开销估算**：~0.3G（额外的 BNNeck + 分类器 x3）
- **移植方案**：我们的 PAMS 已经有了 part feature pooling 和 global feature。可以额外加一个 foreground 分支：将所有可见 part 的特征加权聚合得到前景特征，用独立的 BNNeck + 分类器监督。这相当于在我们现有的 part + global 双分支基础上增加一个 foreground 分支。

### 模块 D: SpatialAttn（空间注意力模块）
- 文件位置：`modeling/backbones/cls_hrnet.py` L258-L271
- 功能：简单的空间注意力模块，用于增强前景区域特征：
  1. Conv2d(256, 1, 3, stride=2, padding=1) + ReLU
  2. Bilinear upsample 恢复原始分辨率
  3. Conv2d(1, 1, 1) — scaling conv
  4. Sigmoid → 空间注意力权重
  - 应用方式：feature_map = feature_map * spatial_attention
- 输入：[B, 256, H, W]
- 输出：[B, 1, H, W]（注意力权重，0-1）
- 依赖：无
- **移植到我们框架的可行性**：高
- **额外显存开销估算**：<0.05G（两个极小的卷积层）
- **移植方案**：在 Swin-Tiny 最后一层输出（reshape 为 [B, 768, 12, 4]）上应用空间注意力，抑制背景区域。需要将 256-dim 改为 768-dim 输入，或在降维后使用。这与 Instruct-ReID 的 MaskModule 异曲同工，但更轻量。

### 模块 E: ARM（Aligned ReID Matching）— 可见性感知匹配
- 文件位置：`utils/reid_metric.py` L12-L67（`R1_mAP_arm`）和 `data/datasets/eval_reid.py` L67-L118（`arm_eval_func`）
- 功能：测试时的可见性感知特征匹配策略（与我们 PAMS 的 visibility-aware distance 类似）：
  1. 对每个图片，通过 part_pred 的 argmax 确定每个像素属于哪个部件
  2. 统计每个部件是否存在（visible_part [B, K-1] 二值向量）
  3. 匹配时：
     - 对 part feature：只计算 query 和 gallery 共同可见的部件的余弦距离
     - 对 global feature：正常计算余弦距离
     - 最终距离 = (part_cosine_sum + global_cosine) / (shared_visible_count + 1)
  4. 距离归一化：cosine 从 [-1,1] 映射到 [0,1]
- 输入：query/gallery 的 part_feat [K-1, 256], g_f_feat [D], visible_part [K-1]
- 输出：similarity score
- **移植到我们框架的可行性**：已实现（我们的 PAMS 已有类似的 visibility-aware distance）
- **额外显存开销估算**：0
- **移植方案**：对比 ISP 的 ARM 和我们的实现：
  - ISP：binary visibility（0/1），基于 argmax 的硬判断
  - 我们 PAMS：soft visibility score（0-1），基于关键点置信度
  - 我们的方案更优，因为 soft score 能处理部分可见的情况
  - 但 ISP 的 "距离归一化" 方式（除以共享部件数+1）值得参考

### 模块 F: Cluster Loss
- 文件位置：`layers/cluster_loss.py` L8-L101（ClusterLoss）和 L104-L248（ClusterLoss_local）
- 功能：聚类损失，约束同 ID 特征的最大类内距离小于不同 ID 特征的最小类间距离：
  ```
  cluster_loss = mean(relu(max_intra_dist - min_inter_dist + margin))
  ```
  - ClusterLoss：基于全局特征
  - ClusterLoss_local：基于局部特征，使用 "shortest distance"（动态时间规整 DTW）计算局部对齐距离
  - 默认 margin=10
- **移植到我们框架的可行性**：中
- **额外显存开销估算**：0（纯计算）
- **移植方案**：可替换或补充 triplet loss。但需要注意 ClusterLoss 的计算复杂度较高（需要计算所有 ID 的中心特征和类内/类间距离），在大 batch 时可能较慢。

## 损失函数

ISP 使用了丰富的损失组合：

```python
loss = xent(cls_score_global, target) +     # Global CE
       xent(cls_score_fore, target) +        # Foreground CE
       xent(cls_score_part, target) +        # Part CE
       0.1 * parsing_criterion(part_pd_score, part_target) +  # Parsing Loss (pixel-level CE)
       triplet(y_global, target)[0] +        # Global Triplet
       triplet(y_fore, target)[0] +          # Foreground Triplet
       triplet(y_part, target)[0]            # Part Triplet
# 如果 IF_WITH_CENTER == 'on'，额外加：
       + 0.0005 * center_loss(y_global) +    # Global Center
       + 0.0005 * center_loss(y_fore) +      # Foreground Center
       + 0.0005 * center_loss(y_part)        # Part Center
```

关键数值：
- Parsing Loss 权重：0.1
- Center Loss 权重：0.0005
- Triplet margin：0.3
- Label smoothing：开启

**可直接用于我们框架的**：
- Parsing Loss（配合伪标签）：权重 0.1
- 三分支独立监督的范式

## 训练 Tricks

- **Backbone**: HRNet-W32（多分辨率特征融合），ImageNet 预训练
- **输入分辨率**: [256, 128]（Occluded-Duke 使用 [384, 128] 可能更好，ISP 代码默认 256）
- **优化器**: Adam, lr=3.5e-4, weight_decay=5e-4
- **调度**: Step LR，milestones=[40, 70]，gamma=0.1
- **Warmup**: linear warmup，10 iterations，factor=0.01
- **总 Epochs**: 120
- **BatchSize**: 64 (16 IDs * 4 instances)
- **聚类周期**: 每 2 个 epoch 重新聚类一次，直到 epoch 101
- **部件数**: 默认 7 类（1 背景 + 6 前景部件），Occluded-Duke 用 6 类（1 bg + 5 parts）
- **数据增强**: Random Horizontal Flip (p=0.5), Random Erasing (p=0.5), Padding=10
- **评估**: ARM 模式（可见性感知匹配），epoch 40 开始每 40 epoch 评估
- **前景增强**: CLUSTERING.ENHANCED=True 时，用 sigmoid 函数 `1/(1+exp(-5*(2x-1)-3))` 增强前景/背景的对比度

## 对我们框架的改进建议

1. **Parsing Loss 用于强化 Part Assignment**（优先级：高）：
   - 利用我们已有的关键点信息生成像素级伪标签（比 ISP 的 K-means 更可靠）
   - 在 Swin-Tiny 输出上加一个轻量分割头（1x1 conv, 768 -> K），用 Parsing Loss 监督
   - 这迫使 backbone 学到更清晰的部件边界特征，提升 part feature 质量
   - 生成伪标签的方法：对每个关键点，基于其坐标在特征图上创建高斯热图，通过 argmax 得到每个像素的 part assignment
   - 预期效果：在 Occluded-Duke 上，明确的部件边界有助于区分可见/不可见区域
   - 显存开销：<0.1G

2. **Foreground 分支**（优先级：中高）：
   - 在现有 part + global 基础上加一个 foreground 分支
   - 前景特征 = 所有可见 part 特征的加权平均（权重为 visibility score）
   - 用独立 BNNeck + 分类器监督
   - 这提供了一个介于 global 和 part 之间的中间粒度特征
   - 对严重遮挡场景，foreground 特征可能比 global 更鲁棒（排除了背景/被遮挡区域）
   - 显存开销：~0.1G（一个 BNNeck + 一个分类器）

3. **SpatialAttn 空间注意力**（优先级：中）：
   - 超轻量的空间注意力模块（2个小卷积）
   - 在 part pooling 之前应用，增强前景区域
   - 可以替代或补充关键点 visibility 信息
   - 显存开销：<0.05G

4. **Identity-Guided Clustering（简化版）**（优先级：低）：
   - ISP 的离线聚类可以作为 PAMS 的初始化手段
   - 在训练前用预训练的 Swin-Tiny 提取特征，做 per-ID 聚类得到部件标签
   - 但我们有关键点信息，聚类的必要性不大

5. **ARM 距离归一化策略**（优先级：中）：
   - 对比我们当前的 visibility-aware distance 和 ISP 的 ARM 公式
   - ISP: `score = (sum_visible_cosine + global_cosine) / (num_visible + 1)`
   - 我们可以对比两种归一化方式，选择效果更好的

6. **不建议移植的模块**：
   - HRNet backbone（我们已锁定 Swin-Tiny）
   - 完整的在线聚类更新流程（工程复杂度高，我们有关键点替代）
   - ClusterLoss_local（DTW 计算复杂度高，batch size 64 时可能较慢）
   - Center Loss（额外优化器复杂度，收益有限）
