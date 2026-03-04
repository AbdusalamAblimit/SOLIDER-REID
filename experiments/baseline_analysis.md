# Baseline 代码分析: SOLIDER-REID with Swin-Tiny

## 1. 整体架构

SOLIDER-REID 是基于 TransReID 框架修改的 ReID 系统，使用 SOLIDER 预训练的 Swin Transformer 作为 backbone。

### 核心文件
- `model/make_model.py`: 模型构建入口，注册 backbone factory
- `model/backbones/swin_transformer.py`: Swin Transformer 实现（L1069-L1408）
- `model/backbones/pams.py`: PAMS 架构（最新实验方向）
- `model/backbones/pose_swin_transformer.py`: 早期 Pose-Swin 双分支实现
- `model/backbones/sptrans.py`: SPTrans 实验
- `model/backbones/pgtdrop.py`: PGTDrop 实验
- `loss/make_loss.py`: 损失函数配置
- `loss/part_loss.py`: Part-Averaged Triplet Loss, Push Loss
- `processor/processor.py`: 训练/推理主循环
- `config/defaults.py`: 全部配置项定义

## 2. Swin-Tiny Backbone

### 结构
- `embed_dims=96, depths=(2,2,6,2), num_heads=(3,6,12,24), window_size=7`
- 4 stages，输出 channels: `[96, 192, 384, 768]`
- Patch size=4, stride=4 (非重叠 patch embedding)
- 输入 `[384, 128]` → Stage 0: `[96, 32]`, Stage 1: `[48, 16]`, Stage 2: `[24, 8]`, Stage 3: `[12, 4]`

### with_cp (梯度检查点)
- 在 `SwinBlockSequence` 的每个 `SwinBlock` 中生效
- 通过 `torch.utils.checkpoint.checkpoint` 减少中间激活的显存
- 配置: `cfg.MODEL.WITH_CP = True`

### SOLIDER 语义控制
- 每个 stage 后有 `semantic_embed_w` 和 `semantic_embed_b` 线性层
- `semantic_weight` 控制 semantic vs appearance 的比例
- `x = x * softplus(sw) + sb`，用于控制特征的语义强度

### 预训练权重加载
- `PRETRAIN_CHOICE: 'self'` → 直接加载完整模型参数
- 预训练文件: `pretrained/swin_tiny.pth`（SOLIDER 预训练）

## 3. 已有模型变体

### 3.1 Standard (swin_tiny)
- 单分支，全局特征，BNNeck + 分类器
- Baseline: 55.2% mAP / 65.5% R-1 on Occ-Duke

### 3.2 Pose-Swin 双分支 (pose_swin_tiny)
- 在 FUSE_STAGE 分叉为 global + local 分支
- 通过姿态热图在 local 分支做 feature fusion (mul/add/gate)
- 使用 MMPose ViTPose 在线推理获取热图
- 最好结果: 59.0% concat mAP on Occ-Duke
- **发现**: 主要提升来自双分支架构(参数翻倍)，pose gating 仅贡献 +0.3%

### 3.3 PAMS (Part-Aware Multi-Scale)
- 单分支 Swin + Multi-Scale Fusion + 可学习 Part Classifier
- 姿态仅用于 BPA (Body Part Attention) 监督，推理时不需要姿态
- 组件:
  - MultiScaleFusion: 融合4个 stage 输出到统一空间分辨率 (24×8)
  - PartClassifier: 1×1 Conv → K+1 (5 parts + background)
  - BPA Loss: 用 pose 热图生成的伪标签监督 part classifier
  - Part-Averaged Triplet Loss: L2 归一化 + 可见性感知距离 + 软间隔
  - Push Loss: 部件特征多样性正则化
- **经历**: v1-v7 全部因 triplet loss 爆炸终止；v8 通过 L2归一化+软间隔解决
- v8 at epoch 30: 45.8% global mAP (仍在训练中，未完成)

## 4. 训练配置

| 参数 | 值 |
|------|-----|
| Optimizer | SGD |
| Base LR | 0.0008 |
| Warmup | 20 epochs, cosine |
| Total Epochs | 120 |
| Batch Size | 64 |
| Input Size | 384 × 128 |
| Label Smooth | off |
| Triplet | Soft Margin (NO_MARGIN: True) |
| AMP | 自动混合精度 (amp.autocast) |

## 5. 评估流程

- R1_mAP_eval 支持多分支评估（global/local/concat/parts）
- 支持 visibility-aware part distance 计算
- ENABLED_FEATS 配置决定参与评估的分支
- 评估间隔: 每 10 个 epoch

## 6. 数据集: Occluded-Duke

- 唯一实验数据集
- 训练集 + 测试集（query + gallery）
- 遮挡场景为主，姿态引导方法的核心测试场景

## 7. 关键发现（from previous experiments）

1. **双分支 vs 姿态**: 双分支带来的 +3.5% mAP 主要来自参数翻倍，pose gating 只贡献 +0.3%
2. **PAMS 思路正确但工程挑战大**: Part 特征的 triplet loss 需要 L2 归一化 + 软间隔才能稳定训练
3. **BPA 监督有效**: 可学习 Part Classifier 比固定 pose 热图更灵活
4. **推理时不需要姿态**: PAMS 的 part classifier 在训练后可以独立工作

## 8. 当前状态

- `dev` 分支包含所有实验代码（PAMS, SPTrans, PGTDrop, Pose-Swin）
- PAMS v8/v9 是最新的工作方向
- 核心问题：PAMS 虽然训练稳定了，但 mAP 是否能超过 59.0% (Pose-Swin best) 尚未验证
