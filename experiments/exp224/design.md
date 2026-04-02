# 实验 exp224: KAMP — Keypoint-Anchored Multi-Scale Part Features (Tiny)

## 动机

### 当前局限
GCN 只从 Stage 3 output (12x4, 768ch) 采样 keypoint features。
但不同 body parts 需要不同尺度的特征：
- 面部/手部: 需要 Stage 1-2 的高分辨率细节
- 躯干/腿部: Stage 3 的语义特征已足够
- **遮挡区域**: 高分辨率特征可以提供更精确的边界信息

### 与已有尝试的区别
- exp096 (MRKF multi-resolution): 尝试融合 Stage 2+3 特征 → 失败
  - 原因: 全图 feature map 级融合太粗糙
- **KAMP**: 只在 keypoint 位置采样各阶段特征，精确且高效
  - 每个 keypoint 从 4 个 stage 的 feature map 分别采样
  - 学习 per-keypoint 的 scale attention (哪个 stage 更重要)

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
