# Paper 5: PGFA — Pose-Guided Feature Alignment for Occluded Person Re-Identification
**来源**: ICCV 2019
**仓库**: https://github.com/lightas/ICCV19_Pose_Guided_Occluded_Person_ReID
**arXiv 摘要**: PGFA proposes a dual-branch architecture with a pose-guided global feature branch (using keypoint heatmaps to attend to visible body regions) and a partial feature branch (horizontal strips), combined with a shared-region distance metric that only compares mutually visible parts between query and gallery.

## 代码架构概览

核心文件结构:
```
ICCV19_Pose_Guided_Occluded_Person_ReID/
├── model.py                    — PCB backbone (ResNet50) + ClassBlock (FC classifiers)
├── train.py                    — 训练: 双分支 (Partial + Pose-Guided Global)
├── test.py                     — 推理: 特征提取 + shared-region evaluate
├── shared_region_evaluate.py   — 核心评测: 可见区域匹配距离计算
├── utils/part_label.py         — 关键点 → 部件可见性标签生成
├── utils/prepare.py            — 数据预处理 (Duke → Occluded-Duke)
├── AlphaPose/                  — AlphaPose 姿态估计 (离线提取)
│   ├── generate_heatmap.py     — 关键点 → Gaussian heatmap 生成
│   └── heatmap.py              — Gaussian heatmap 核函数
└── dataset/                    — Occluded-Duke 数据集转换工具
```

模型入口: `train.py` 中 `train_model()` 函数
- Backbone: ResNet50 (去掉 layer4 下采样, stride=1)
- 双分支: Partial Feature Branch + Pose-Guided Global Feature Branch
- 推理时使用 `shared_region_evaluate.py` 中的 shared-region distance

**关键创新**: 这是 Occluded ReID 的开山之作, 提出了:
1. Occluded-DukeMTMC 数据集
2. 姿态引导的特征对齐范式
3. Shared-region distance metric

## 可拆解模块清单

### 模块 A: Pose-Guided Global Feature Branch — 姿态热图加权特征聚合
- 文件位置: `/root/work/paper_repos/ICCV19_Pose_Guided_Occluded_Person_ReID/train.py` L167-L183
- 功能: 用 18 个关键点的 Gaussian 热图对 CNN 特征做空间加权, 然后 MaxPool 聚合为全局特征
- 输入:
  - features: [B, 2048, H', W'] (ResNet50 layer4 输出, H'=24, W'=8 for 384x128 input)
  - masks: [B, 18, 24, 8] (18 个关键点的 Gaussian 热图, resize 到特征图尺寸)
- 输出: pg_global_feature [B, 4096]
- 内部结构:
  ```python
  # 1. 全局平均池化: features → [B, 2048] (无姿态引导的 baseline 特征)
  pg_global_feature_1 = AdaptiveAvgPool2d(1,1)(features)  # [B, 2048]

  # 2. 逐关键点加权: 对 18 个关键点分别做 element-wise 乘法 + 池化
  for i in range(18):
      mask = masks[:, i, :, :]           # [B, 24, 8]
      mask = mask.unsqueeze(1).expand_as(features)  # [B, 2048, 24, 8]
      pg_feature = mask * features       # 空间加权
      pg_feature = AdaptiveAvgPool2d(1,1)(pg_feature)  # [B, 2048, 1]
      # 收集 18 个加权特征

  # 3. MaxPool 聚合: [B, 2048, 18] → MaxPool1d → [B, 2048]
  pg_global_feature_2 = AdaptiveMaxPool1d(1)(all_pose_features)

  # 4. 拼接: [B, 2048] (global) + [B, 2048] (pose-weighted) = [B, 4096]
  pg_global_feature = cat(pg_global_feature_1, pg_global_feature_2)
  ```
- 依赖: 离线预计算的 18-channel heatmap (来自 AlphaPose)
- **移植到我们框架的可行性**: 中-高
  - 核心思想简单且通用: 用姿态热图对特征做空间加权
  - 需要将 ResNet-2048-d 适配为 Swin-Tiny 768-d
  - 热图需要 resize 到 Swin 最终 feature map 尺寸 (12x4 for 384x128 with patch_size=4, 4 stages)
- **额外显存开销估算**: ~0.1-0.2G
  - 主要是热图存储 [B, 18, H', W'] 和 18 次 element-wise 乘法
  - 非常轻量, 几乎不影响显存
- **移植方案**:
  1. 离线提取关键点 + 生成 Gaussian 热图, 存为 .npy
  2. DataLoader 加载热图, resize 到 Swin 最终 feature map 尺寸
  3. 在 Swin 最后一个 stage 的输出 feature map 上执行 element-wise 加权
  4. 输出维度: 768 (avg) + 768 (pose-weighted max) = 1536 → 可降维或直接拼接
  5. **适配 Swin 的关键点**: Swin 输出是 [B, N, 768] 的 token 序列, 需要先 reshape 成 [B, 768, H', W'] 的空间 feature map, 然后才能做空间加权

### 模块 B: Partial Feature Branch — 水平条带特征
- 文件位置: `/root/work/paper_repos/ICCV19_Pose_Guided_Occluded_Person_ReID/train.py` L157-L166
- 功能: 将特征图水平切分为 K 个条带, 每个条带独立提取特征并分类
- 输入: features [B, 2048, 24, 8] (backbone 输出)
- 输出: K 个 [B, 2048] 的 part 特征
- 内部结构:
  ```python
  # 水平分割: AdaptiveAvgPool2d(K, 1) → [B, 2048, K, 1] → squeeze
  partial_feature = AdaptiveAvgPool2d((part_num, 1))(features)  # [B, 2048, K]
  # 每个 part 通过独立 ClassBlock 分类
  for i in range(K):
      output = PCB_classifier[i](partial_feature[:, :, i])
      loss = CE(output, labels)
  ```
- 依赖: 无
- **移植到我们框架的可行性**: 低 (我们已有 PAMS 的部件特征, 功能重复)
  - PGFA 的水平条带是简单均匀划分, 不如 PAMS 的 part-aware 方式好
  - 不建议移植此模块

### 模块 C: Shared-Region Distance — 可见区域匹配距离
- 文件位置: `/root/work/paper_repos/ICCV19_Pose_Guided_Occluded_Person_ReID/shared_region_evaluate.py` L9-L60
- 功能: 测试时仅在 query 和 gallery 共同可见的身体区域上计算距离, 忽略被遮挡的部分
- 输入:
  - qf, gf: partial features [N_part, D] per sample
  - qpl, gpl: part labels (binary, 1=visible, 0=occluded) [N_part]
  - qf2, gf2: pose-guided global features [D]
- 输出: 相似度分数
- 内部结构:
  ```python
  # 1. 计算 pose-guided global distance (cosine)
  s2 = cosine(q_global, g_global)      # 全局相似度
  s2 = (s2 + 1) / 2                     # [-1,1] → [0,1]

  # 2. 计算 shared-region partial distance
  overlap = gpl * qpl                   # 共同可见区域: element-wise AND
  s = cosine(q_partial, g_partial)      # 逐 part 相似度
  s = (s + 1) / 2                       # [-1,1] → [0,1]
  s = s * overlap                       # 只保留共同可见 parts 的距离

  # 3. 加权平均
  final_score = (s.sum() + s2) / (overlap.sum() + 1)
  ```
- 依赖: 需要 query 和 gallery 图像的关键点信息来生成 part visibility labels
- **移植到我们框架的可行性**: 中
  - 思想非常好: 遮挡场景下只在共同可见区域做匹配
  - 但需要在 inference 时有 pose 信息来生成 part labels (增加推理流程复杂度)
  - 与我们 PAMS 的 part_vis (可见性分数) 有协同潜力
- **额外显存开销估算**: 可忽略 (纯推理时的距离计算)
- **移植方案**:
  1. 利用 PAMS 的 part_vis 替代 PGFA 的 binary part label
  2. 在评测时, 用 part_vis 作为加权系数: `score = sum(vis_q * vis_g * cosine(part_q, part_g)) / sum(vis_q * vis_g)`
  3. 这比 PGFA 的 binary overlap 更精细 (连续可见性分数 vs 0/1)

### 模块 D: Part Label Generation — 关键点到部件可见性标签
- 文件位置: `/root/work/paper_repos/ICCV19_Pose_Guided_Occluded_Person_ReID/utils/part_label.py` L8-L43
- 功能: 根据关键点 y 坐标判断哪些水平条带是可见的
- 输入: 关键点 JSON (18 keypoints with x, y, confidence), part_num, image height
- 输出: [part_num] 的 binary 数组 (1=该条带有可见关键点)
- 内部结构:
  ```python
  # 对每个检测到的关键点 (confidence > 0.2):
  #   确定它在图像高度方向属于哪个 part strip
  #   标记该 strip 为可见 (label[k] = 1)
  for keypoint in detected_keypoints:
      y = keypoint.y
      for k in range(part_num):
          if y > (k/part_num)*img_h and y < ((k+1)/part_num)*img_h:
              label[k] = 1
  ```
- 依赖: AlphaPose 离线提取的关键点 JSON
- **移植到我们框架的可行性**: 中
  - 简单但实用的可见性判断方法
  - 可作为 PAMS 可见性预测的辅助监督信号
- **移植方案**: 用离线关键点生成 part visibility labels, 作为 PAMS part_vis 的 GT

### 模块 E: Gaussian Heatmap Generation — 关键点热图生成
- 文件位置: `/root/work/paper_repos/ICCV19_Pose_Guided_Occluded_Person_ReID/AlphaPose/heatmap.py` L22-L30
  和 `/root/work/paper_repos/ICCV19_Pose_Guided_Occluded_Person_ReID/AlphaPose/generate_heatmap.py`
- 功能: 以每个关键点为中心生成 2D Gaussian 热图, 然后 resize 到 feature map 尺寸
- 输入: 关键点坐标 (x, y), 图像尺寸, variance
- 输出: [18, H_feat, W_feat] 热图数组 (值域 [0, 1])
- 生成流程:
  ```python
  # 1. 对每个关键点 (x, y), 在原图尺寸上生成 Gaussian:
  #    G(p) = exp(-(px-x)^2 + (py-y)^2) / (2*var^2))
  #    var = sqrt(img_h * img_w / 1000)
  # 2. 将 [img_h, img_w] 的热图 resize 到 [24, 8] (feature map size)
  # 3. 保存为 .npy 文件, 每张图一个 [18, 24, 8] 数组
  ```
- 依赖: 关键点坐标 (来自 AlphaPose)
- **移植到我们框架的可行性**: 高
  - 纯预处理步骤, 不影响模型
  - 可直接使用, 只需调整 resize 目标尺寸为我们 Swin feature map 的 spatial size
  - 对于 [384, 128] 输入 + Swin-Tiny: 最终 feature map = [12, 4] (经过 4 stages 的 2x 下采样)
- **移植方案**:
  1. 用现有关键点数据 (JSON) 生成 [18, 12, 4] 或 [17, 12, 4] 热图
  2. 保存为预处理 .npy 文件
  3. DataLoader 加载并传入模型

## 损失函数

### 训练损失
- **CrossEntropy Loss** (标准, 无 label smoothing)
- 双分支加权:
  ```python
  loss = lambda * PCBloss + (1 - lambda) * PG_global_loss
  # lambda = 0.2 (默认)
  # PCBloss = sum of K part CE losses
  # PG_global_loss = CE on pose-guided global feature
  ```
- 无 Triplet Loss, 无 Center Loss (2019 年的做法, 相对简单)

### 评测距离
- **Shared-Region Cosine Distance**: 只在共同可见 parts 上计算 cosine similarity
- 距离公式: `score = (sum(overlap * part_cosine) + global_cosine) / (num_visible_parts + 1)`

## 训练 Tricks

1. **Backbone**: ResNet50 (ImageNet pretrained), 去掉 layer4 的 stride=2 下采样
2. **输入尺寸**: [384, 128] — 与我们一致
3. **优化器**: SGD, lr=0.1 (backbone lr * 0.1), weight_decay=5e-4, momentum=0.9, nesterov=True
4. **LR 调度**: StepLR, step_size=40, gamma=0.1
5. **总 epochs**: 60 (很短)
6. **Batch size**: 32
7. **Part 数量**: K=3 (默认, 可调至 6)
8. **ClassBlock 结构**: Linear(D→256) → BN → LeakyReLU(0.1) → Dropout(0.5) → Linear(256→num_classes)
9. **数据增强**: Random horizontal flip (prob=0.5), 同步翻转热图
10. **离线姿态提取**: AlphaPose, 18 keypoints per image
11. **热图 sigma**: `sqrt(img_h * img_w / 1000)` — 自适应图像尺寸

## 对我们框架的改进建议

### 建议 1: Pose Heatmap Spatial Attention (高优先级)
- **核心思路**: 将 PGFA 的热图加权方式适配到 Swin-Tiny
- **实现**:
  1. 将 Swin 最后一层的 token 序列 [B, N, 768] reshape 为 [B, 768, H', W']
  2. 加载预计算的 pose heatmap [B, 18, H', W']
  3. 对 18 个 heatmap 分别做 element-wise 加权 + AdaptiveAvgPool2d(1,1)
  4. AdaptiveMaxPool1d(1) 聚合 18 个 pose-weighted features → [B, 768]
  5. 与 global avg pool feature 拼接 → [B, 1536] 或降维到 [B, 768]
- **与 PAMS 的协同**: PAMS 的 part features 是语义级别的部件; pose heatmap attention 是空间级别的。两者互补。
- **注意**: 需要将热图 resize 到 Swin 的 feature map 尺寸 (对于 [384,128] 输入约为 [12,4])

### 建议 2: Visibility-Weighted Evaluation Distance (中优先级)
- **核心思路**: 测试时用可见性信息加权距离计算
- **与 PAMS 结合**: PAMS 已经预测 part_vis, 可以直接用:
  ```python
  # query: part_feats [K, D], part_vis [K]
  # gallery: part_feats [K, D], part_vis [K]
  vis_weight = query_vis * gallery_vis  # 共同可见性
  part_sim = cosine(query_parts, gallery_parts)  # [K]
  score = (vis_weight * part_sim).sum() / (vis_weight.sum() + eps)
  ```
- **不需要修改模型**, 只需修改评测逻辑

### 建议 3: Part Visibility Labels 作为监督信号 (中优先级)
- **核心思路**: 用 PGFA 的 part_label_generate 方法从关键点生成 GT visibility labels
- 用这些 labels 监督 PAMS 的 part_vis 预测 (目前 PAMS 的 part_vis 来自 BPA attention, 可能不够准确)
- **实现**: 离线生成 visibility labels → 作为 BPA 模块的额外 BCE 监督

### 注意事项
- PGFA 的方法比较早期 (2019), 很多设计被后续工作改进 (如 PFD, KPR)
- 但其核心思想 — 只在可见区域做特征匹配 — 仍然是 occluded ReID 最重要的理念
- pose heatmap 加权比简单的水平条带划分更好, 因为它是空间自适应的
- 热图方法需要预处理, 但代码简单且计算量极低
- PGFA 的 Shared-Region Distance 思想与我们 PAMS 的 part_vis 天然契合
