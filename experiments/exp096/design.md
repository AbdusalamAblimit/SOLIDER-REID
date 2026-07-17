# 实验 exp096: MRKF (Multi-Resolution Keypoint Features)

## 动机
- exp095 DPF 失败：热图空间池化在 12×4 分辨率下比点采样更差 (-1.6%)
- **根本原因已确认**：Stage 3 feature map 仅 12×4 = 48 个空间位置，分辨率太低
- KPR (ECCV24) 通过 FPN 获得 stride-4 特征（96×32=3072 位置），大幅提升 per-part 精度
- **本实验**：不做完整 FPN（太复杂），而是简单地从 Stage 2 (24×8, 384-dim) 额外采样关键点特征

## 核心想法
在两个尺度上采样关键点特征并融合：
- **Stage 3 (12×4, 768-dim)**：高语义、低分辨率 → 捕获身份级特征
- **Stage 2 (24×8, 384-dim)**：中语义、中分辨率 → 4× 更精确的空间采样

GCN 在多尺度融合特征上传播，能同时利用精确空间信息和语义身份信息。

## 技术方案

### 数据流
```
Backbone:
  Stage 0 (96d, 96×32) → Stage 1 (192d, 48×16) → Stage 2 (384d, 24×8) → Stage 3 (768d, 12×4)
                                                        ↓ [保存]                ↓ [正常输出]
                                                        ↓                       ↓
                                              bilinear sample @kp        bilinear sample @kp
                                              kp_s2: (B, 17, 384)      kp_s3: (B, 17, 768)
                                                        ↓                       ↓
                                              Linear(384→256)           Linear(768→256)
                                                        ↓                       ↓
                                                        └──── concat ────┘
                                                              ↓
                                                    kp_fused: (B, 17, 512)
                                                              ↓
                                                    Linear(512→768) [融合投影]
                                                              ↓
                                                    GCN (768-dim, 2 layers)
                                                              ↓
                                                    Weighted pool → BN → ID loss
```

### 修改文件
1. **`model/pose_backbone_model.py`**:
   - `_run_backbone_with_psg()`: 保存 Stage 2 输出（需要 LayerNorm + reshape）
   - 返回值中包含 stage2_featmap
   - `forward()`: 传递 stage2_featmap 给 skeleton_head

2. **`model/modules/skeleton_gcn.py`**:
   - `SkeletonGCNHead.__init__()`: 新增 s2_proj (384→256) + s3_proj (768→256) + fusion_proj (512→768)
   - 新增 `_multires_sample_features()`: 在两个尺度上采样并融合
   - `forward()`: 当 MRKF 启用且 stage2_featmap 可用时使用多尺度采样

3. **`config/defaults.py`**: POSE_MRKF = False

4. **`configs/occluded_duke/pose_psg_gcn_paa_mrkf.yml`**

### 参数估算
- s2_proj: Linear(384→256) = 98K
- s3_proj: Linear(768→256) = 197K
- fusion_proj: Linear(512→768) = 394K
- **总计: ~690K 新参数**

### 预期结果
- Stage 2 的 4× 分辨率优势应该显著改善 per-keypoint 特征质量
- 预期 mAP +0.5~1.5% over exp066 (PAA baseline)
- 如果成功，证明分辨率确实是 GCN 分支的瓶颈

### 对照组
- Baseline: exp066 (PSG+GCN+PAA), 61.6%/74.2% @Ep120
- 消融变量：仅改变 keypoint feature extraction（从 Stage 3 单尺度 → Stage 2+3 多尺度）

### 风险
- Stage 2 特征可能语义不够强（较浅的网络层）
- 需要从 backbone forward 中额外提取 Stage 2 输出（改动较大）
- 690K 新参数是否足够学习好的尺度融合
