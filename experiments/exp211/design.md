# 实验 exp211: GCN+PAA+CE+OA-SD + MaxSim Triplet (MST)

## 动机
- MaxSim hybrid test-time matching 给 +1.7% mAP (70.6→72.3)
- 但模型的 per-keypoint features 是为 CE+triplet 训练的，不是为 MaxSim 优化的
- 如果直接用 MaxSim 距离做 triplet loss，per-keypoint features 会直接为 MaxSim matching 优化
- PKC (SupCon on keypoints) 不改变 equal_concat 性能，但可能对 MaxSim 有帮助 — MST 更直接

## 核心假设
MaxSim Triplet loss 直接优化 per-keypoint features 在 MaxSim 距离度量下的判别力。
这比 PKC (SupCon) 更加 aligned with test-time evaluation metric。

## 技术方案

### MaxSim Triplet Loss
```python
# For each anchor i in batch:
#   Find hardest positive (same ID, max MaxSim distance)
#   Find hardest negative (diff ID, min MaxSim distance)
#   loss = max(0, d_pos - d_neg + margin)
# Where d = MaxSim distance between two sets of 17 keypoint features
```

### MaxSim 距离计算
```python
# kp_feats: (B, 17, C)
# For two samples (i, j):
#   sim(i,j) = mean_k max_l cos(kp_i_k, kp_j_l)  # MaxSim
#   dist(i,j) = 1 - sim(i,j)
```

### 修改文件
- `config/defaults.py`: POSE_MST, POSE_MST_WEIGHT, POSE_MST_MARGIN
- `processor/processor.py`: MaxSim triplet loss computation

## 对照组
- exp206r (no MST, no PKC): 70.6/82.6 (eq) → 72.3/82.9 (maxsim)
- exp210b (PKC=0.05): ~70.5/82.0 (eq) → TBD (maxsim)
