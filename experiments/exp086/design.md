# 实验 exp086: Pose-Aware Parallel Augmentation Training (PA-PAT)

## 动机
- PADE (ICASSP 2024) 证明三路并行增强训练对遮挡 ReID 有效
- 我们的 ROA 已证明遮挡增强有效 (+1.27%)
- **新想法**: 结合 PADE 的三路训练和我们的 pose 框架
- 三路增强: 原图 + ROA遮挡 + Pose-Guided Erasing
- 三路共享 backbone，分别计算 loss

## 创新点
- PADE 用 random crop + random erasing 模拟遮挡
- 我们改进: 用 **pose-guided erasing** 替代 random erasing
  - 按身体部位擦除（有 pose 指导，更真实）
  - 与 ROA（外部物体遮挡）形成互补
- 三路训练 = 更多样化的遮挡模式

## 技术方案

### 三路数据
1. **view_full**: 标准增强（当前 pipeline）
2. **view_roa**: ROA 遮挡增强（已有）
3. **view_pge**: Pose-Guided Body Part Erasing（擦除随机身体部位）

### 训练流程
```
For each batch:
  img_full, img_roa, img_pge, pose, label = batch

  # 三次 forward (共享 backbone)
  score1, feat1, ... = model(img_full, pose)
  score2, feat2, ... = model(img_roa, pose)
  score3, feat3, ... = model(img_pge, pose)

  # Loss 聚合
  loss = (loss_fn(score1, feat1, label) +
          loss_fn(score2, feat2, label) +
          loss_fn(score3, feat3, label)) / 3
```

### 实现改动
1. `datasets/pose_dataset.py`: 返回 3 个增强版本
2. `processor/processor.py`: 3 次 forward + loss 聚合
3. 不改模型代码

### Pose-Guided Body Part Erasing
- 随机选择 1-2 个身体部位 (5 parts)
- 用 pose heatmap 定位该部位的空间区域
- 将该区域像素置零（或随机噪声）
- 类似 exp016 PGE 但更温和（只擦 1-2 个部位，不是全部）

## 参数
- 零额外参数
- 训练时间 ~3x（但可以减少 epoch 到 60-80）

## 风险
- 训练时间 3x 可能不值得
- pose-guided erasing 在 exp016 曾有害（但那是擦整个身体，这里只擦 1-2 部位）

## 对照
- exp066 PAA = 61.6%/74.2%
- exp079 ROA = 62.0%/73.6%
