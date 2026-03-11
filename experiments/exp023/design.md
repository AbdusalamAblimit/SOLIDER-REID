# 实验 exp023: Market-1501 PSG (Pose Spatial Gate)

## 动机
- PSG 在 Occluded-Duke 上取得了 +1.7% mAP 的最佳结果 (exp007)
- 需要验证 PSG 在非遮挡数据集 Market-1501 上是否同样有效
- 跨数据集泛化性是论文的重要证据之一

## 创新点 / 核心想法
- 核心假设：PSG 的 pose spatial gate 机制在非遮挡场景下也能通过关注人体区域来提升性能
- 如果 PSG 在 Market-1501 上也有效，说明 pose spatial gating 是一种通用的 ReID 改进策略
- 如果无效，说明 PSG 主要解决遮挡问题，创新点的 story 需要聚焦遮挡场景

## 技术方案
- 与 Occluded-Duke exp007 完全相同的 PSG 架构
- Config: `configs/market/pose_backbone_psg.yml`
- PSG: 2 gates in Stage 3, Conv2d(17→64→768), sigmoid, zero-init
- 需要先完成 Market-1501 pose 数据提取
- Output: `./log/market1501/exp023_psg/`

## 预期结果
- 如果 PSG 有效：mAP +0.5~1.0%（非遮挡数据集增益通常较小）
- 如果 PSG 中性/有害：mAP ±0.2%，说明 PSG 的优势主要来自遮挡处理

## 对照组
- Baseline 对照：exp022 Market-1501 baseline
- 消融变量：仅增加 PSG（POSE_BACKBONE_PSG=True）
