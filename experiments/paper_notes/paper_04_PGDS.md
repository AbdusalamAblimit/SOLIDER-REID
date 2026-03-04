# Paper 4: PGDS — Pose-Guided Deep Supervision for Mitigating Clothes-Changing in Person Re-Identification
**来源**: AVSS 2024 (Oral)
**仓库**: https://github.com/huyquoctrinh/PGDS
**arXiv 摘要**: PGDS uses a frozen OpenPose encoder to distill pose-structural knowledge into a Swin-Tiny ReID backbone via multi-layer KL divergence, enabling the human encoder to learn body-part awareness without extra inference cost.

## 代码架构概览

核心文件结构:
```
PGDS/
├── model/make_model.py        — 主模型定义 (build_transformer + TTK modules)
├── cloth_changing.py          — PoseSolider 融合模型 (实验性代码,未在主训练中使用)
├── pytorchOpenpose/src/model.py — OpenPose body pose model + body_pose_reid wrapper
├── processor/custom_processor.py — 训练循环: KL divergence distillation logic
├── processor/fusion.py        — Fusion module (attention-weighted bilinear pooling, unused)
├── loss/make_loss.py          — 标准 ID+Triplet loss 构建
├── loss/koleo_loss.py         — Kozachenko-Leonenko entropy regularizer
├── configs/market/swin_tiny.yml — 默认配置
├── train.py                   — 入口: 加载 ReID model + frozen pose model
└── test.py                    — 推理: 仅使用 ReID model (无 pose model)
```

模型入口: `train.py` → `make_model()` 构建 `build_transformer` (Swin-Tiny backbone), 同时加载冻结的 `body_pose_reid` (OpenPose backbone)

**关键设计**: PGDS 的核心思想是「训练时用姿态知识蒸馏,推理时完全丢弃姿态模型」。这意味着推理阶段 zero overhead。

### 架构流程

1. **Human Encoder** (`build_transformer`): Swin-Tiny backbone, 输出 global_feat (768-d) + 3层中间特征 (通过 TTK 模块投影到 768-d)
2. **Pose Encoder** (`body_pose_reid`): 冻结的 OpenPose → flatten → Linear(768) → L2 norm, 输出 768-d pose embedding
3. **PHP (Pose-to-Human Projection)**: 通过 KL 散度将 pose embedding 对齐到 3 个中间层 TTK 输出

## 可拆解模块清单

### 模块 A: TTK (Transformer Token Knowledge) — 多层特征投影模块
- 文件位置: `/root/work/paper_repos/PGDS/model/make_model.py` L156-L188
- 功能: 将 Swin-Tiny 各 stage 的 2D feature map 投影为固定维度 (768-d) 的 token，用于与 pose embedding 做 KL 对齐
- 输入:
  - TTK1: [B, 96, 96, 32] (Stage 0 output)
  - TTK2: [B, 192, 48, 16] (Stage 1 output)
  - TTK3: [B, 384, 24, 8] (Stage 2 output)
- 输出: [B, 768] (L2-normalized)
- 内部结构:
  ```python
  # 1. Reshape: [B, C, W, H] → [B, C, W*H]
  # 2. Channel reduction: [B, W*H, C] → Linear(C → num_head=16) → [B, W*H, 16]
  # 3. Transpose: [B, 16, W*H]
  # 4. Spatial projection: [B, 16, W*H] → Linear(W*H → 768) → [B, 16, 768]
  # 5. Transformer encoder layer (1 layer, d_model=768, nhead=8)
  # 6. Mean pooling over 16 tokens → [B, 768]
  # 7. L2 normalize
  ```
- 参数量估算:
  - TTK1: Linear(96→16) + Linear(96*32=3072→768) + TransformerEncoderLayer(768, 8heads) ≈ 7.1M
  - TTK2: Linear(192→16) + Linear(48*16=768→768) + TransformerEncoderLayer ≈ 7.1M
  - TTK3: Linear(384→16) + Linear(24*8=192→768) + TransformerEncoderLayer ≈ 7.1M
  - 总计: ~21.3M params (3个TTK), 但仅训练时使用
- 依赖: 无外部依赖, 仅需 Swin 中间 feature maps
- **移植到我们框架的可行性**: 高
  - 我们的 Swin-Tiny 同样产出 4 个 stage 的 feature maps (96/192/384/768-d channels)
  - 可直接接入 stage 0-2 的输出
  - 训练时使用,推理时可完全移除,零额外推理开销
- **额外显存开销估算**: ~0.5-0.8G
  - 3 个 TTK 模块各含 1 个 TransformerEncoderLayer(768), 参数量约 21M
  - 但中间激活值是主要开销; 由于 TTK 只在训练时使用, 可用 gradient checkpointing 控制
  - OpenPose 模型冻结,不需要梯度,显存占用较小 (~0.3G forward only)
- **移植方案**:
  1. 修改 Swin-Tiny backbone 的 forward 方法,返回中间 stage 的 feature maps (已有 `out_indices` 支持)
  2. 适配我们框架的 stage dimensions:
     - 对于 [384,128] 输入: Stage0=[B,96,96,32], Stage1=[B,192,48,16], Stage2=[B,384,24,8]
     - TTK 参数需要匹配这些尺寸 (PGDS 的配置已经匹配 [384,128] 输入)
  3. 可进一步精简: 只用 2 个 TTK (Stage 1+2) 而非 3 个以节省显存

### 模块 B: body_pose_reid — 姿态编码器
- 文件位置: `/root/work/paper_repos/PGDS/pytorchOpenpose/src/model.py` L142-L162
- 功能: 将 OpenPose backbone 的 PAF (Part Affinity Field) 输出转换为 768-d 姿态 embedding
- 输入: [B, 3, 384, 128] (原始图像)
- 输出: score [B, 767], pose_feat [B, 768] (L2-normalized)
- 内部结构:
  ```python
  # 1. OpenPose forward → pose_feat [B, 38, H', W'] (PAF maps, 38 channels)
  # 2. Mean over channel dim → [B, H'*W'] → flatten
  # 3. Linear(768) → [B, 768]
  # 4. Cls head: Linear(768 → num_classes)
  # 5. L2 normalize pose_feat
  ```
- 依赖: OpenPose 预训练权重 (body_pose_model.pth, ~25MB)
- **移植到我们框架的可行性**: 中
  - OpenPose 模型本身比较重 (~25M params), 但完全冻结不需要梯度
  - 可替换为更轻量的姿态模型 (如 lightweight OpenPose 或 MoveNet)
  - 或者完全换成离线提取的 pose features, 预计算后存入文件
- **额外显存开销估算**: ~0.3G (冻结模型 forward only, 无梯度)
- **移植方案**:
  1. 方案 A (在线): 直接在训练时加载冻结 OpenPose, 与 PGDS 一致
  2. 方案 B (离线, 推荐): 用现有姿态模型 (HRNet/DWPose) 离线提取关键点 → 转换为固定维度 embedding → 存为 .npy → DataLoader 加载
  3. 方案 B 更省显存, 但需要预处理流水线

### 模块 C: KL Divergence Multi-Layer Distillation — 核心训练策略
- 文件位置: `/root/work/paper_repos/PGDS/processor/custom_processor.py` L79-L91
- 功能: 用 KL 散度将冻结姿态编码器的输出蒸馏到 ReID backbone 的多个中间层
- 实现细节:
  ```python
  # pose_feat: [B, 768] from frozen pose encoder
  # local_feat1/2/3: [B, 768] from TTK1/2/3 on Swin stages 0/1/2

  div_loss1 = KLDiv(softmax(pose_feat), softmax(local_feat1)) / 2
  div_loss2 = KLDiv(softmax(pose_feat), softmax(local_feat2)) / 2
  div_loss3 = KLDiv(softmax(pose_feat), softmax(local_feat3)) / 2

  total_loss = 0.8 * reid_loss + 0.2 * (0.5*div_loss1 + 0.3*div_loss2 + 0.2*div_loss3)
  ```
- 层级权重分配: 浅层 (Stage 0) 权重最高 (0.5), 深层 (Stage 2) 最低 (0.2)
  - 设计逻辑: 浅层更接近空间/结构信息, 与姿态的空间结构对应更紧密
- **移植到我们框架的可行性**: 高
  - 纯 loss 层面的改动, 不影响模型结构
  - 只需修改训练循环添加 KL loss 项
  - 推理时完全不涉及
- **额外显存开销估算**: 可忽略 (仅额外 softmax + KL 计算)
- **移植方案**:
  1. 修改训练循环, 在计算 reid_loss 后追加 KL divergence loss
  2. 权重比例可作为超参数调节 (PGDS 用 0.8:0.2 分配给 reid_loss:pose_distill_loss)
  3. 层级权重 (0.5/0.3/0.2) 也可调节

### 模块 D: Fusion — Attention-Weighted Bilinear Pooling (未使用)
- 文件位置: `/root/work/paper_repos/PGDS/processor/fusion.py` L20-L45
- 功能: 用 attention maps 对 feature maps 做 bilinear pooling
- 注意: 此模块在 `custom_processor.py` 中被 import 但被注释掉了 (`# fusion = Fusion(768).to(local_rank)`)
- 架构: Conv2d(2048→M=8 attention heads) + einsum(attention, features) + signed sqrt + L2 norm + Linear(8*2048→768) + BN
- **移植到我们框架的可行性**: 低 (被原作者弃用, 设计针对 ResNet-2048-d, 不适合 Swin)

## 损失函数

### 1. ReID Loss (标准, 权重 0.8)
- **Label Smoothing CE Loss** + **Soft Triplet Loss** (no margin)
- ID Loss 权重: `cfg.MODEL.ID_LOSS_WEIGHT` (默认 1.0)
- Triplet Loss 权重: `cfg.MODEL.TRIPLET_LOSS_WEIGHT` (默认 1.0), L2 normalize features

### 2. Pose Distillation Loss (权重 0.2)
- **KL Divergence** (`nn.KLDivLoss(reduction='batchmean')`)
- 三层加权: 0.5 * KL(stage0) + 0.3 * KL(stage1) + 0.2 * KL(stage2)
- 各层 KL 除以 2 (对称化)

### 3. 其他 Loss (代码中存在但未使用)
- **KoLeo Loss**: Kozachenko-Leonenko entropy regularizer, 鼓励特征空间均匀分布
- **Center Loss**: 缩小类内距离
- **ArcFace Loss**: 在代码中导入但未在主训练循环中使用
- **SupCon Loss**: Supervised Contrastive, 导入但未使用

## 训练 Tricks

1. **优化器**: SGD, base_lr=0.0002, weight_decay=1e-4, bias_lr_factor=2
2. **调度器**: Cosine Annealing with warmup (5 epochs)
3. **总 epochs**: 250 (相当长)
4. **Batch size**: 64, NUM_INSTANCE=2
5. **混合精度**: 使用 `torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)` 和 GradScaler
6. **冻结姿态模型**: `for param in pose_reid.parameters(): param.requires_grad = False`
7. **Label Smoothing**: 开启
8. **Triplet Loss**: Soft margin (NO_MARGIN=True), L2 normalize features (TRP_L2=True)
9. **SOLIDER 预训练**: 使用 SOLIDER 的 Swin-Tiny 权重作为初始化, semantic_weight=0.2
10. **输入尺寸**: [384, 128], 与我们的 baseline 完全一致
11. **数据增强**: Random horizontal flip (prob=0.5), Random erasing (prob=0.5), Padding=10
12. **推理时零开销**: pose model 和 TTK 模块在推理时完全不参与

## 对我们框架的改进建议

### 建议 1: 多层姿态知识蒸馏 (高优先级)
- **核心思路**: 在训练时用冻结的姿态编码器蒸馏到 Swin-Tiny 的中间层
- **优势**: 推理时零额外开销, 但模型已隐式学会关注身体结构
- **实现路径**:
  1. 离线提取 pose features (用现有 HRNet/DWPose), 存为 768-d embedding per image
  2. 在 DataLoader 中加载 precomputed pose embedding
  3. 添加 TTK 模块到 Swin stage 0-2 的输出
  4. 训练时加 KL divergence loss, 推理时移除 TTK 和 pose data

### 建议 2: 简化版 — 单层 Pose Distillation
- 只在 Swin 最后一层 (Stage 3, 768-d) 做 KL 蒸馏, 省略 TTK 模块
- 直接用 global_feat (768-d) 与 pose_embedding (768-d) 做 KL divergence
- 零额外参数, 零额外推理开销, 极其轻量
- 缺点: 可能不如多层蒸馏效果好

### 建议 3: 结合我们的 PAMS 部件特征
- PAMS 已经有 part-level features; 可以对每个 part feature 分别做姿态蒸馏
- 例如: head part feature ← 头部关键点 embedding, torso part ← 躯干关键点 embedding
- 这比 PGDS 的全局 pose embedding 蒸馏更精细

### 注意事项
- PGDS 的 TTK 模块参数量不小 (~21M for 3 TTKs), 训练时显存占用约 0.5-0.8G
- 如果显存紧张, 优先考虑建议 2 (单层蒸馏) 或只保留 TTK2+TTK3 (去掉最大的 Stage 0 TTK)
- 可将 OpenPose 替换为我们的离线姿态数据以避免在线推理姿态模型的显存开销
