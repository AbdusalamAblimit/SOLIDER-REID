# 实验 exp030: PDS+StopGrad + Skeleton GCN Part Branch

## 动机
- 29 个实验证明：Part 分支的热图 mask 池化产生的特征与 Global 高度冗余
- exp028 证明 Part 收敛不是瓶颈 — 即使完全收敛也无测试增益
- **根本原因**：热图 mask 池化只是在相同的特征图上做不同空间加权，没有引入新的结构信息
- **新方向**：用 Skeleton GCN 在人体骨架拓扑上传播特征，创造基于**结构关系**的表征
- 灵感来源：动作识别领域的 ST-GCN，Tran-GCN (IET 2025)

## 创新点 / 核心想法
- **核心假设**：人体骨架拓扑提供了空间池化无法捕获的结构信息。当某个关节点被遮挡时，GCN 可以从相邻可见关节传播特征，实现结构化特征补全。
- **与前序实验的区别**：
  - Part Pooling（exp001-008）：在 feature map 上做空间加权 → 结果与 GAP 冗余
  - GCN：在 17 个关键点的图结构上传播特征 → 引入拓扑先验

## 技术方案

### 新增模块：`model/modules/skeleton_gcn.py`
- SkeletonGCN: 2 层图卷积网络
- 节点：17 个 COCO 关键点
- 边：COCO 骨架定义的 16 条骨骼连接 + 自环
- 维度：768 → 256 → 768（residual connection）
- 邻接矩阵：D^{-1/2} A D^{-1/2} 归一化（固定，不可学习）
- 参数量：768×256 + 256×768 ≈ 400K

### 数据流
```
Input Image → Swin Stage 0-2 (shared)
              ↓
    ┌─────────┴─────────┐
    ↓ (clone)           ↓ (detach / stop_grad)
  Global Stage 3        Part Stage 3
  + PSG gates           (no PSG)
    ↓                   ↓
  GAP → global_feat   Feature Map (B, 768, 12, 4)
                        ↓
                      Bilinear Sample @ 17 keypoints
                        ↓
                      (B, 17, 768) keypoint features
                        ↓
                      Skeleton GCN (2 layers)
                        ↓
                      (B, 17, 768) enhanced features
                        ↓
                      Confidence-weighted average
                        ↓
                      skeleton_feat (B, 768)
                        ↓
                      BN + Classifier (1 ID + 1 Triplet)
```

### 关键点坐标映射
- pose_dict 中 keypoints 是原始图像像素坐标 (17, 2)
- 映射到 feature map: x' = x / img_w * feat_w, y' = y / img_h * feat_h
- 使用 F.grid_sample 进行双线性采样（需归一化到 [-1, 1]）
- 使用 person 0（主要人物）的关键点

### 置信度加权
- 每个关键点的 score 作为权重
- skeleton_feat = sum(score_i * feat_i) / sum(score_i)
- 低置信度（遮挡）关键点自动降权

### 配置选项
- `POSE_SKELETON_GCN: True` — 启用 Skeleton GCN 替代 Part Pooling
- `POSE_GCN_LAYERS: 2` — GCN 层数
- `POSE_GCN_HIDDEN: 256` — GCN 隐藏层维度

## 预期结果
- **乐观**：GCN 的骨架拓扑传播让 Part 特征真正互补 → concat > global-only (59.5%+)
- **中性**：GCN 特征与 Global 仍然冗余 → concat ≈ global-only (~59%)
- **悲观**：稀疏关键点采样丢失信息 → 低于 exp023 (~58%)

## 对照组
- exp023 (PDS+StopGrad, Part Pooling): global-only 59.5%, concat_scaled 59.1%
- 消融变量：Part 分支特征提取方式（Part Pooling → Skeleton GCN）
