# 实验 exp224: 2-Stage Keypoint Feature Fusion (Tiny)

## 动机

### 当前局限
GCN 只从 Swin 最后一层 output (12x4, 768ch) 采样 keypoint features。
倒数第二层 (24x8, 384ch) 有更高空间分辨率，可能对精确的 keypoint 定位有帮助。

### 与已有尝试的区别
- exp096 (MRKF multi-resolution): 尝试融合 2 层特征 → 失败
  - 原因: 拼接+投影太粗糙
- **本实验**: 只在 keypoint 位置采样两层特征，用 learned per-keypoint scale attention 融合
  - 每个 keypoint 独立选择两层特征的加权比

### 创新门槛
1. ✅ **问题层面**: 重新定义 "part features 应该从哪个尺度提取"
2. ✅ **机制层面**: Per-keypoint learned scale selection 在 ReID 中是新的
3. ✅ **证据层面**: 直接消融各 stage 的贡献

## 技术方案

### 实现
在 SkeletonGCNHead 中，不只从 Stage 3 采样，而是从所有 stages 采样并融合：

```python
# Stage outputs from backbone:
# outs[0]: (B, 96, 96, 32)   -- Stage 1, high-res
# outs[1]: (B, 192, 48, 16)  -- Stage 2
# outs[2]: (B, 384, 24, 8)   -- Stage 3  
# outs[3]: (B, 768, 12, 4)   -- Stage 4 (current only)

# For each keypoint, sample from all stages:
for stage_fm in stage_outputs:
    kp_feat_s = grid_sample(stage_fm, keypoint_coords)  # project to common dim

# Fuse with learned per-keypoint scale attention:
scale_weights = scale_attn(kp_feat_stage4)  # (B, 17, 4) -- per-keypoint per-stage
fused_kp_feat = sum(scale_weights * per_stage_feats)
```

### 修改文件
- `model/modules/skeleton_gcn.py`: 新增 multi-scale sampling + fusion
- `model/pose_backbone_model.py`: 传递所有 stage outputs 给 GCN head
- `config/defaults.py`: POSE_MULTI_SCALE_KP

## 对照组
- exp191 OA-SD (single stage): 63.2/75.4
- exp220 GSPB + MaxSim: 64.6 (current Tiny best mAP)
