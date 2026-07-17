# 实验 exp233: Per-Body-Part Independent Training (KPR-Inspired)

## 动机

**KPR (ECCV 2024) = 75.1% mAP on Occluded-Duke** — 当前无 reranking SOTA。
KPR 的核心与我们系统的最大差异: **per-body-part 独立训练**。

当前系统: 17 GCN keypoints → weighted pool → 1 skeleton_feat → 1 classifier + 1 triplet
KPR 做法: keypoints → 6 body part groups → 6 independent classifiers + 6 triplet losses

**为什么这可能有效**:
1. 独立 classifier 迫使每个 body part 学习 discriminative features
2. 不同 body part 不会互相稀释（当前 pooling 把所有信息压成 1 vector）
3. 在 detached features 上操作 — 无 backbone 梯度干扰
4. Test time 可以用 per-part distance aggregation (更精细的 matching)

## 核心假设

6-body-part 独立训练让每个 part 特征更具判别力，提升 Part branch 整体贡献。

## 技术方案

### COCO 17 关键点 → 6 body parts 分组
```python
BODY_PART_GROUPS = {
    'head':     [0, 1, 2, 3, 4],      # nose, eyes, ears
    'torso':    [5, 6, 11, 12],        # shoulders, hips
    'left_arm': [5, 7, 9],             # l_shoulder, l_elbow, l_wrist
    'right_arm':[6, 8, 10],            # r_shoulder, r_elbow, r_wrist
    'left_leg': [11, 13, 15],          # l_hip, l_knee, l_ankle
    'right_leg':[12, 14, 16],          # r_hip, r_knee, r_ankle
}
```

### 修改
1. **skeleton_gcn.py**: GCN 后不做 single pool，而是 per-group weighted pool → 6 features (B, 6, C)
2. **pose_backbone_model.py**: 6 个独立 BN + classifier，返回 [global, part1...part6] 的 scores 和 feats
3. **make_loss.py**: 自动处理 7-element list (global + 6 parts)，每个 part 有独立 ID + triplet loss
4. **test**: 可以用 equal_concat (global + 6 parts) 或 per-part distance aggregation

### Config
- `MODEL.POSE_GCN_PER_PART True`
- No new weights needed beyond 6 × (BN + classifier) ≈ 6 × 1.5K = 9K extra params

## 对照组
- exp191 (OA-SD, pooled GCN): 63.2/75.4

## 预期结果
- 成功: +1.0~2.0% mAP (per-part discrimination 改善 Part 特征)
- 失败: ~0% (detached features 信息有限，分开训练不增加信息)

## 早停
- ep10 < 25% → 终止
