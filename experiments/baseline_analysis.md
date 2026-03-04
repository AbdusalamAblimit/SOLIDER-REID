# Baseline Analysis: SOLIDER-REID Framework

## 1. 总体架构

SOLIDER-REID 是基于 SOLIDER 预训练的 ReID 系统，支持多种 backbone（ResNet、ViT、Swin Transformer），我们使用 **Swin-Tiny** 配置。

### 目录结构
```
├── train.py / test.py          → 训练/测试入口
├── config/defaults.py          → YACS 配置定义（~280行）
├── configs/occluded_duke/      → Occluded-Duke 实验配置
├── model/
│   ├── make_model.py           → 模型工厂（3种架构：Backbone, build_transformer, build_transformer_local）
│   └── backbones/
│       ├── swin_transformer.py → Swin backbone（~1400行）
│       ├── pose_swin_transformer.py → 姿态引导双分支 Swin
│       ├── pams.py             → Part-Aware Multi-Scale 架构
│       ├── sptrans.py          → Semantic-Pose Trans
│       └── pgtdrop.py          → Pose-Guided Token Dropping
├── loss/
│   ├── make_loss.py            → Loss 工厂（支持多策略）
│   ├── triplet_loss.py         → Hard mining triplet loss
│   ├── softmax_loss.py         → Label smoothing CE
│   └── part_loss.py            → Part-averaged triplet + push loss
├── datasets/
│   ├── make_dataloader.py      → DataLoader 工厂
│   └── occluded_duke.py        → Occluded-Duke 数据集定义
├── processor/processor.py      → 训练循环 & 推理逻辑
├── solver/                     → 优化器 & 学习率调度器
└── utils/metrics.py            → R1_mAP_eval（含 visibility-aware 距离）
```

## 2. Swin-Tiny Backbone 详解

**文件**: `model/backbones/swin_transformer.py`

### 架构参数
- **Embed dim**: 96 → 各 stage 翻倍 [96, 192, 384, 768]
- **Depths**: [2, 2, 6, 2]（共 12 个 Transformer block）
- **Num heads**: [3, 6, 12, 24]
- **Window size**: 7
- **Patch size**: 4（non-overlapping）
- **Input**: [384, 128] → patch 后 [96, 32] → 3072 tokens
- **Output**: stage 3 特征 768 维

### WITH_CP 梯度检查点
在 `SwinBlock.forward()` 中实现：
```python
if self.with_cp and x.requires_grad:
    x = cp.checkpoint(_inner_forward, x)
else:
    x = _inner_forward(x)
```
- 效果：显存从 ~12GB 降至 ~7-8GB，速度慢 ~20%
- 通过 config `WITH_CP: True` 开启，传入 Swin 构造函数

### SOLIDER 预训练权重加载
1. `make_model.py` L194: `factory[cfg.MODEL.TRANSFORMER_TYPE](..., pretrained=model_path)`
2. `swin_transformer.py` `init_weights()`: 使用 mmengine 加载 checkpoint
3. 自动处理 `module.` 前缀、shape 不匹配（插值 position embedding）

## 3. 特征提取流程

### 标准路径（swin_tiny）
```
Image [B, 3, 384, 128]
  → Swin-Tiny backbone → global_feat [B, 768]
  → BNNeck(768) → feat_bn [B, 768]
  → Training: classifier(feat_bn) → logits [B, num_classes]
  → Eval: return global_feat (NECK_FEAT='before')
```

### 双分支路径（pose_swin_tiny / sptrans / pgtdrop）
```
Image → Swin stages 0..branch_stage → 共享特征
  → Global branch: stages branch_stage+1..3 → global_feat [B, D]
  → Local branch: 姿态引导特征 → local_feat [B, D]
  → 各自 BNNeck + classifier
  → Eval: {global, local, concat} 三种特征
```

### PAMS 路径
```
Image → Swin stages 0..2 → 多尺度特征
  → Multi-Scale Fusion → [B, H*W, D]
  → Body Part Attention (pose heatmap) → part_feats [B, K, D]
  → Global pooling → global_feat, foreground_feat
  → K 个 part BNNeck + classifier
  → Eval: {global, parts + visibility} → visibility-aware distance
```

## 4. Loss 函数

### 标准配置
- **ID Loss**: CrossEntropyLabelSmooth（label smoothing=0.1）或 plain CE（IF_LABELSMOOTH='off'）
- **Triplet Loss**: Soft margin triplet（NO_MARGIN=True → SoftMarginLoss），hard example mining
- **权重**: ID=1.0, Triplet=1.0

### PAMS 配置（loss strategy='pams'）
- ID Loss: CE(global) + CE(foreground), weight=1.0
- Part ID Loss: per-part CE averaged, weight=0.5
- Triplet: PartAveragedTripletLoss（visibility-weighted）, weight=1.0
- BPA Loss: Body Part Attention CE, weight=1.0
- Push Loss: 部件多样性正则, weight=0.1

## 5. 训练配置

### 核心超参数（Occluded-Duke, swin_tiny）
| 参数 | 值 |
|------|-----|
| Batch Size | 64 (16 ID × 4 samples) |
| Optimizer | SGD |
| Base LR | 0.0008 |
| Warmup | 20 epochs, cosine |
| Total Epochs | 120 |
| Weight Decay | 1e-4 |
| Input Size | [384, 128] |
| Random Erasing | prob=0.5 |
| Eval Period | 每 10 epoch |

### AMP 混合精度
`processor.py` 中默认开启 `amp.autocast(enabled=True)`，使用 GradScaler。

## 6. 评估方式

**文件**: `utils/metrics.py` - `R1_mAP_eval`

- **距离度量**: 标准路径用 Euclidean distance（L2 normalized features）
- **PAMS 路径**: visibility-aware distance（仅在 query 和 gallery 都可见的部件上计算距离）
- **协议**: 去除同 camera 的 gallery 样本（standard ReID protocol）
- **指标**: mAP, Rank-1, Rank-5, Rank-10

## 7. 已有实验结果

### Occluded-Duke（主实验数据集）
| 方法 | mAP | Rank-1 |
|------|-----|--------|
| Baseline (swin_tiny, 无姿态) | 55.2% | 65.5% |
| + Pose dual-branch (E1) | 59.0% (concat) | 68.6% |
| + Dual-branch no-pose (E7) | 58.7% (concat) | 68.8% |

### 关键发现
1. 双分支架构比姿态引导本身贡献更大（+3.5% vs +0.3%）
2. 姿态信息的融合方式（multiplicative gating）效果有限
3. PAMS 架构已集成但未充分评估

## 8. 改进空间分析

### 当前瓶颈
1. **姿态信息利用不充分**：当前的 multiplicative gating 只带来 +0.3% mAP
2. **Part-based 特征**：PAMS 的 part attention 机制可能需要更好的姿态热图
3. **遮挡处理**：visibility-aware distance 是正确方向，但需要更精准的可见性估计

### 改进方向
1. 从 KPR/PFD 等论文学习更有效的姿态-特征融合方式
2. 离线提取高质量关键点/热图（HRNet-W48 / DWPose / PifPaf）
3. 设计 part-aware 的特征提取和匹配策略
4. 改进 loss 函数（part-specific loss, visibility-guided loss）
