# 实验 exp217: Occlusion-Equivariant Representation Learning (OERL) — Tiny

## 动机

### 当前问题
所有方法（包括我们的 PSG+GCN+OA-SD）都在"增强鲁棒性"——通过数据增强或蒸馏让模型对遮挡不敏感。
但没有任何方法教模型**理解遮挡的结构**：遮挡左臂 vs 遮挡下半身对特征有什么不同的影响？

### 核心洞察
训练集 95.8% 可见，几乎所有图片都能看到完整身体。我们可以利用这一点：
- 对同一图片施加不同的 pose-guided 遮挡（左半身、右半身、上半身、下半身等）
- 让模型学习**遮挡模式与特征变化之间的对应关系**
- 同时约束：可见部分的 per-part features 不受其他部分遮挡的影响

### 与现有方法的本质区别
| 方法 | 做什么 | 问题 |
|------|--------|------|
| PLBOA | 加遮挡增强数据 | 只是更多样本，没学到遮挡结构 |
| OA-SD | teacher clean → student occluded distillation | 只对齐结果，不理解变换 |
| **OERL** | **学习 clean→occluded 的特征变换 + 可见部分不变性** | **理解遮挡的几何学** |

## 核心假设
1. **Part Invariance**: 如果 part k 在 clean 和 occluded 中都可见，part k 的 token 应该不变
2. **Equivariance** (简化版): 不同遮挡模式对 global feature 的影响是可预测的

## 技术方案

### Phase 1: Part Occlusion Invariance (POI) — 本实验
最简核心：**可见部分对其他部分的遮挡不变**

```python
# 1. Forward clean image → get per-part features (from GCN keypoint sampling)
clean_kp_feats = sample_keypoints(featmap_clean, keypoints)  # (B, 17, C)

# 2. Apply pose-guided partial occlusion (random subset of body parts masked)
img_occ, occ_mask = random_pose_occlusion(img, heatmaps)
# occ_mask: (B, 17) binary — which keypoints are occluded

# 3. Forward occluded image → get per-part features
occ_kp_feats = sample_keypoints(featmap_occ, keypoints)  # (B, 17, C)

# 4. Part Invariance Loss: visible parts should match
visible = ~occ_mask & (kp_scores > 0.3)  # (B, 17) — visible in BOTH views
poi_loss = cosine_distance(clean_kp_feats[visible], occ_kp_feats[visible]).mean()
```

### 关键区别 vs OA-SD
- OA-SD: distill **global** feature, 用 **EMA teacher** (不同权重)
- POI: 对齐 **per-part** features, 用 **同一模型** (clean vs occluded forward)
- POI 的梯度从两个 forward 同时回传到 backbone → 双重优化信号

### 遮挡模式
不只是 PLBOA (下半身)，而是多种 pose-guided 遮挡：
1. 左半身 (left shoulder, elbow, wrist, hip, knee, ankle)
2. 右半身 (right side)
3. 上半身 (above hip)
4. 下半身 (below hip) — 类似 PLBOA
5. 随机 40-60% keypoints

### 修改文件
- `config/defaults.py`: POSE_OERL, POSE_OERL_WEIGHT
- `datasets/pose_dataset.py`: 新增多样化 pose-guided 遮挡
- `processor/processor.py`: POI loss 计算（两次 forward, per-part cosine 对齐）
- `model/pose_backbone_model.py`: 不需要修改

### 显存
需要两次 forward (clean + occluded) → ~2x 显存。Tiny on 3090 (6GB * 2 = 12GB) 可行。

## 预期结果
- 假设成立: mAP +2-3% on Tiny (64.9 → 67-68%)
- 如果 POI 有效 → 加 equivariance predictor (Phase 2)

## 对照组
- exp030a Tiny baseline: 60.7%
- exp191 Tiny + OA-SD: 64.4%
- exp187 Tiny + SupCon 3-view: 64.9%
- exp217 Tiny + OERL: 目标 67%+
