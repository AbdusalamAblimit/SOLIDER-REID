# 实验 exp157: Pose-Guided Lower-Body Occlusion Augmentation (PGLBOA)

## 动机

训练/测试遮挡分布 gap 分析：
- 训练集下半身 <50% vis: 1.8%
- Query 集下半身 <50% vis: **24.4%** (13x gap!)
- 最常被遮挡：ankle(42%), knee(27%), wrist(31%)

ROA 用随机位置贴 VOC 物体，不针对下半身。我们需要**针对性地增强下半身遮挡**。

## 技术方案

对每张训练图，以概率 p=0.5：
1. 从 pose keypoints 中取下半身关键点（hip, knee, ankle）的 y 坐标
2. 在 hip 以下的区域应用 rectangular erasing（用随机颜色或灰色填充）
3. 遮挡区域高度：从 hip_y 到图底部的 30-70%

实现：在 datasets/pose_dataset.py 的 __getitem__ 中，在 standard RE 之前或之后。

## 为什么这不同于 ROA
- ROA 贴 VOC 物体到随机位置 → 均匀增强
- PGLBOA 只遮挡下半身 → 针对性弥补 train-test gap

## 预期
- 应比 ROA 更有效（ROA +1.0%，PGLBOA 可能 +0.5~1.5%）
- 两者可能正交叠加
