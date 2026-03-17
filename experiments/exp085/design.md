# 实验 exp085: Parallel Augmentation Training (PAT)

## 动机
- PADE (ICASSP 2024) 的核心创新: 三路并行增强训练
- 同一图片的三个版本同时训练: 原图 + crop遮挡 + mandatory erasing
- 我们已有 ROA (+1.27%)，但 ROA 是"叠加外部遮挡物"
- PADE 的 crop 是"随机裁剪模拟遮挡"——更简单但可能同样有效
- **假设**: 多样化的遮挡增强训练比单一 ROA 更有效

## 创新点 / 核心想法
- 不是加新模块，而是改训练范式
- 每个 batch 的每张图片用三种不同增强同时训练:
  1. img_full: 标准增强 (现有 pipeline)
  2. img_crop: RandomResizedCrop 模拟遮挡
  3. img_erase: 强制 100% Random Erasing
- 三个版本用同一 backbone 分别前向传播
- 三套 loss 加权平均

## 技术方案
- 修改 dataloader: 每个样本返回 3 个增强版本
- 修改 processor: 3 次 forward + loss 聚合
- 不改模型架构
- 训练时间 ~3x（但 120 epoch 变 40 epoch 等效？）

## 参数
- 零额外参数（共享 backbone）
- 但训练时间 ~3x

## 注意
- pose data 需要与 crop/erase 对齐（crop 改变了图片内容）
- crop 后的图片不一定包含目标人物——需要小心处理

## 对照
- exp066 PAA = 61.6%/74.2%
- exp079 ROA = 62.0%/73.6%
- exp067 PAA+ROA = 62.0%/73.7%
