# 实验 exp097: LPS (Learned Part Segmentation)

## 动机
- exp095 DPF 失败：热图空间池化在 12×4 分辨率下不如点采样 (-1.6%)
- exp096 MRKF 负面：多尺度融合也不帮助 (-2.1% @Ep40，趋势持续恶化)
- **但 KPR 在类似分辨率下成功了** → 为什么？
- **KPR 的关键不同**：使用 **learned per-pixel body-part classifier**（1×1 Conv2d → softmax）
  而非 raw heatmap。分类器从 768-dim 特征内容判断每个位置属于哪个身体部位，
  即使在 12×4 的分辨率下也足够区分头/躯干/腿等大区域。

## 核心创新
**用可学习的逐像素身体部位分类器替代固定点采样**

关键区别：
- 点采样：在关键点坐标处采一个点 → 精确但脆弱（依赖关键点定位精度）
- LPS：学习每个空间位置属于哪个部位，软注意力池化 → 利用部位的完整空间范围

LPS + Skeleton GCN = **全新的 per-part 特征提取 + 图传播框架**：
1. 分类器学习把 48 个空间位置分配给 K 个部位
2. 每个部位通过注意力池化获得完整的部位级特征
3. GCN 在部位之间传播信息（遮挡恢复）
4. 热图用于监督分类器（不是直接用作注意力权重）

## 技术方案

### 数据流
```
Stage 3 feature map: (B, 768, 12, 4)
          ↓
PixelToPartClassifier: Conv2d(768, K+1, 1×1) → softmax
          ↓
Part probability maps: (B, K+1, 12, 4)  [K body parts + background]
          ↓
Soft attention pooling: feat_k = Σ(prob_k * features) / Σ(prob_k)
          ↓
Per-part features: (B, K, 768)
          ↓
GCN propagation (skeleton edges)
          ↓
Enhanced part features → weighted pool → ID loss + triplet
```

### 分类器监督
- 从 heatmap 生成 ground truth 部位标签：
  - 每个空间位置 (h, w) 的标签 = argmax(heatmap[:, h, w])
  - 如果所有 heatmap 响应都很低 → 标签 = background
- Loss: cross_entropy(pixel_classifier_output, gt_labels)
- Weight: 较小（如 0.1），辅助损失

### 身体部位划分 (K=5)
按 COCO 17 关键点分组：
- Part 0 (Head): keypoints 0-4 (nose, eyes, ears)
- Part 1 (Torso): keypoints 5-6 (shoulders)
- Part 2 (Arms): keypoints 7-10 (elbows, wrists)
- Part 3 (Upper legs): keypoints 11-12, 13-14 (hips, knees)
- Part 4 (Lower legs): keypoints 15-16 (ankles)

### 关键超参数
- K = 5 (身体部位数)
- Part classifier: 1×1 Conv2d(768, 6), ~4.6K 参数
- Segmentation loss weight: 0.1

### 修改文件
1. `model/modules/skeleton_gcn.py` — 重大改动：
   - 新增 PixelToPartClassifier
   - 新增 soft attention pooling (替代 _sample_keypoint_features)
   - 修改 forward 使用 part masks 而非点采样
   - GCN 从 17 关键点 → 5 身体部位（需要新的邻接矩阵）

2. `model/pose_backbone_model.py` — 传递 heatmaps 给 GCN head

3. `config/defaults.py` — POSE_LPS 相关配置

4. `processor/processor.py` — 处理 segmentation loss

## 预期结果
- 如果成功：mAP +1~2% over exp066（学习的部位分割应该比点采样更好）
- 0 额外推理开销（分类器是 1×1 conv，极轻量）
- 如果失败：最可能是 12×4 分辨率下 5 部位分类不够精确

## 对照组
- exp066 (PSG+GCN+PAA, point sampling): 61.6%/74.2%
- 消融：仅改变 keypoint feature extraction 方法

## 论文价值
- 这是 KPR 核心机制的一种变体，但与 skeleton GCN 结合是新的
- 如果成功，说明 learned part segmentation > point sampling 即使在低分辨率下
- 与 SGCFR 结合：learned parts + graph recovery + cross-image recovery = 完整框架
