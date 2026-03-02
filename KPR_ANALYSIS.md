# KPR (Keypoint Promptable Re-Identification) 分析报告

> 论文: Somers et al., "Keypoint Promptable Re-Identification", ECCV 2024
> 仓库: https://github.com/VlSomers/keypoint_promptable_reidentification

---

## 1. KPR 核心架构

```
输入图像 + 可选关键点 Prompt
        │
        ▼
  Backbone (SOLIDER Swin / ViT)
  ┌─ Prompt Tokenizer: 关键点热力图 → patch embedding → 加到图像 token 上
  └─ 输出多尺度空间特征 [N, D, Hf, Wf]
        │
        ▼
  Pixel-to-Part Classifier (可学习注意力)
  → 预测每个像素属于哪个身体部位 → softmax → parts_masks [N, K+1, Hf, Wf]
        │
        ├── Global Embedding:    全局平均池化
        ├── Foreground Embedding: 前景 mask 加权池化
        ├── Background Embedding: 背景 mask 加权池化
        ├── Parts Embeddings:     K 个部位分别加权池化 → [N, K, D]
        └── Concat Parts:        K 个部位特征拼接 → [N, K*D]
        │
        ▼
  各分支独立 BNClassifier → ID 分类 + 特征
```

---

## 2. 与我们 SOLIDER-REID 的关键差异对比

| 维度 | SOLIDER-REID (我们) | KPR |
|------|-------------------|-----|
| **Pose 用法** | 热力图直接乘 feature map (mul/add/gate) | 热力图作为 Prompt Token 加到 patch embedding |
| **特征粒度** | global + local (2 分支) | global + foreground + background + K parts + concat (5 类) |
| **部位划分** | 基于热力图峰值的简单分区 | 可学习 Pixel-to-Part Classifier + softmax |
| **遮挡处理** | 无显式处理 | Visibility Score 机制 (binary/continuous) |
| **距离计算** | 标准欧氏距离 | Visibility-aware 加权距离 (跳过不可见部位) |
| **Loss** | ID loss + Triplet loss (全局) | GiLt: Global-ID + Local-Triplet 分层策略 |
| **负样本 Prompt** | 无 | 遮挡者的关键点作为负样本通道 |
| **Part 监督** | 无 (无监督热力图) | Body Part Attention Loss 显式监督部位分割 |

---

## 3. 可借鉴的核心模块 (按优先级排序)

### 3.1 Visibility Score 机制 ⭐⭐⭐⭐⭐

**问题**: 我们当前对遮挡情况没有任何显式处理。当身体部位被遮挡时，对应的 local feature 是噪声。

**KPR 做法**:
```python
# 从 part attention map 计算每个部位的可见性分数
if binary_visibility:
    # argmax → one-hot → 是否存在该部位的像素
    parts_visibility = one_hot(argmax(parts_probs)).amax(dim=(2,3))  # [N, K] bool
else:
    # 连续值: 该部位最大概率值
    parts_visibility = parts_probs.amax(dim=(2,3))  # [N, K] float [0,1]
```

**可移植性**: 高。我们已有 pose heatmap，可以直接从 heatmap 峰值计算 visibility score。

**移植建议**:
- 在 `pose_swin_transformer.py` 的 forward 中，计算每个关键点热力图的最大值作为 visibility
- 在测试时的距离矩阵计算中，只比较双方都可见的部位

---

### 3.2 GiLt Loss 策略 ⭐⭐⭐⭐⭐

**问题**: 我们对 global 和 local 分支使用相同的 loss 组合，没有区分。

**KPR 做法**:
```python
# Global/Foreground/Concat → 只用 ID Loss (分类)
# Parts → 只用 Triplet Loss (度量学习)
default_weights = {
    GLOBAL:       {'id': 1.0, 'tr': 0.0},
    FOREGROUND:   {'id': 1.0, 'tr': 0.0},
    CONCAT_PARTS: {'id': 1.0, 'tr': 0.0},
    PARTS:        {'id': 0.0, 'tr': 1.0},  # 关键!
}
```

**直觉**:
- 全局特征适合 ID 分类 (softmax 容易收敛)
- 局部部位特征适合 triplet (学习细粒度度量空间)

**移植建议**:
- 修改 `loss/make_loss.py`，对 global branch 只用 ID loss，对 local branch 只用 triplet loss
- 简单改动，无需新增模块

---

### 3.3 可学习 Pixel-to-Part Classifier ⭐⭐⭐⭐

**问题**: 我们的 pose heatmap 来自固定的外部模型 (ViTPose)，无法端到端优化部位分割。

**KPR 做法**:
```python
class PixelToPartClassifier(nn.Module):
    # 1x1 conv: 将 feature map 每个像素分类到 K+1 个部位
    def __init__(self, in_channels, parts_num):
        self.conv = nn.Conv2d(in_channels, parts_num + 1, 1)  # +1 for background

    def forward(self, x):
        return self.conv(x)  # [N, K+1, Hf, Wf]
```
配合 Body Part Attention Loss 用外部人体解析标签监督:
```python
# 用 pose 关键点生成的 GT mask 监督 pixel classifier
loss = CrossEntropyLoss(pixel_cls_scores, gt_part_labels)
```

**移植建议**:
- 在 Swin 最后一个 stage 后加一个 1x1 conv 作为 part classifier
- 用我们已有的 ViTPose 热力图生成伪 GT label 来监督
- 这样 part attention 可以端到端微调

---

### 3.4 Visibility-Aware 距离计算 ⭐⭐⭐⭐

**问题**: 测试时我们的距离矩阵不区分遮挡，被遮挡部位的噪声特征会干扰匹配。

**KPR 做法**:
```python
# 逐部位计算距离
part_distances[k] = euclidean(query_part_k, gallery_part_k)  # [Nq, Ng]

# 只比较双方都可见的部位
valid_mask[k] = vis_query[k] * vis_gallery[k]  # [Nq, Ng]

# 加权平均 (跳过不可见部位)
distance = masked_mean(part_distances, valid_mask)
```

**移植建议**:
- 修改 `utils/metrics.py` 中的评测距离计算
- 保存每个 sample 的 visibility score，测试时使用

---

### 3.5 多分支特征表示 (Foreground + Parts + Concat) ⭐⭐⭐

**问题**: 我们只有 global + local 两个分支。

**KPR 新增分支**:
- **Foreground**: 所有部位 mask 取 max → 前景区域池化 (过滤背景)
- **Concat Parts**: K 个部位特征拼接 → 一个长向量做 ID 分类
- **Background**: 背景区域特征 (辅助训练)

**移植建议**:
- 低优先级。先把 visibility 和 GiLt loss 搞好效果可能已经够了
- 如果要加，foreground embedding 最简单: `fg_mask = heatmaps.max(dim=1); fg_feat = feat * fg_mask`

---

### 3.6 Part-Based Triplet Loss (Visibility-Aware) ⭐⭐⭐

**KPR 做法**:
```python
# 计算 K 个部位的 pairwise distance [K, N, N]
part_dist = pairwise_distance(embeddings)  # [K, N, N]

# Visibility mask: 只有双方该部位都可见时才有效
valid_mask = vis.unsqueeze(1) * vis.unsqueeze(2)  # [K, N, N]

# 平均可见部位距离
combined_dist = masked_mean(part_dist, valid_mask)  # [N, N]

# 标准 batch hard triplet mining
hardest_pos = max_dist(same_id_pairs)
hardest_neg = min_dist(diff_id_pairs)
loss = relu(hardest_pos - hardest_neg + margin)
```

**移植建议**:
- 替换我们现有的 soft triplet loss 为 visibility-aware 版本
- 对 local 分支特别有用

---

### 3.7 Prompt Tokenizer (热力图嵌入方式) ⭐⭐

**问题**: 我们目前的 pose 融合是后融合 (在 feature map 上做 mul/add)。KPR 是前融合。

**KPR 做法**:
```python
# 方式1: embed_heatmaps_patches (默认)
# 热力图 [B, K+2, H, W] → PatchEmbed → part_tokens [B, num_patches, D]
part_tokens = masks_patch_embed(prompt_masks)
image_features += part_tokens  # 直接加到图像 patch embedding 上

# 方式2: spatialize_part_tokens
# 每个部位学一个 token，根据 heatmap 的 argmax 分配到对应空间位置
```

**与我们的区别**:
- 我们: heatmap 在中间层 feature map 上做乘/加 (mid-level fusion)
- KPR: heatmap 在输入层就加到 patch embedding (early fusion)

**移植建议**:
- 可以作为一种新的 `FUSION_MODE` 选项加入
- 但需要改动 backbone 的输入层，工作量较大

---

### 3.8 数据增强: Keypoint Dropout ⭐⭐

**KPR 做法**:
```python
DropRandomKeypoints(p=0.2, ratio=0.5)  # 20% 概率随机丢弃 50% 关键点
DropAllKeypoints(p=0.3)                 # 30% 概率丢弃所有关键点
```

**目的**: 让模型不过度依赖 pose prompt，保持无 pose 时也能工作。

**移植建议**: 在我们的数据增强管线中添加 heatmap dropout。

---

## 4. 推荐实施路线图

### Phase 1: 快速提升 (改动小，收益高)

1. **GiLt Loss 策略**: global 用 ID loss，local 用 triplet loss
2. **Visibility Score**: 从 pose heatmap 计算可见性
3. **Visibility-Aware 距离**: 测试时只比较可见部位

### Phase 2: 中等改动

4. **可学习 Part Classifier**: 在 backbone 后加 1x1 conv + Part Attention Loss
5. **Foreground Embedding**: 从 heatmap 推导前景 mask 做前景池化
6. **Visibility-Aware Triplet**: 训练时的 triplet loss 也考虑可见性

### Phase 3: 深度改造

7. **Prompt Tokenizer 融合方式**: 热力图作为额外 token 输入
8. **Keypoint Dropout 增强**: 增强 prompt 鲁棒性
9. **负样本关键点**: 处理多人遮挡场景

---

## 5. 关键代码文件索引

| 模块 | KPR 路径 |
|------|----------|
| 主模型 | `torchreid/models/kpr.py` |
| Prompt Tokenizer | `torchreid/models/promptable_transformer_backbone.py` |
| SOLIDER 骨干 | `torchreid/models/promptable_solider.py` |
| GiLt Loss | `torchreid/losses/GiLt_loss.py` |
| Part Attention Loss | `torchreid/losses/body_part_attention_loss.py` |
| Part Triplet Loss | `torchreid/losses/part_averaged_triplet_loss.py` |
| 距离计算 | `torchreid/metrics/distance.py` |
| 热力图生成 | `torchreid/utils/imagetools.py` |
| 训练引擎 | `torchreid/engine/image/part_based_engine.py` |
| 默认配置 | `torchreid/scripts/default_config.py` |
