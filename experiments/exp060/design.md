# 实验 exp060: Pose-Aware ROA (PA-ROA)

## 动机
- exp058 ROA (+1.07% mAP) 有效但不新颖（已有多篇论文使用 VOC object pasting）
- 现有 ROA 随机放置遮挡物，不考虑人体位置——遮挡物可能落在纯背景区域，不模拟真实遮挡
- 真实 Occluded-Duke 中的遮挡特点：遮挡物（人/车/栏杆）通常遮挡**身体某部分**
- **PA-ROA 的创新**：用 pose heatmap 引导遮挡物放置位置，只在身体区域上方粘贴

## 创新点 / 核心想法
- **核心假设**: 在身体可见区域（而非随机位置）放置遮挡物，能更好地模拟真实遮挡分布
- **与随机 ROA 的区别**: 随机 ROA 的 center 是 uniform random in [0, img_width]×[0, img_height]。PA-ROA 的 center 从 body keypoint 位置中采样（加随机偏移）
- **创新叙事**: "Pose-aware occlusion simulation" — 第一个用 pose 信息引导遮挡增强位置的方法
- **对照实验**: exp058 (random ROA) vs exp060 (pose-guided ROA)

## 技术方案
- 修改 `occlude_with_objects` 的 center 采样逻辑：
  - 从 person 0 的有效关键点中随机选一个作为遮挡中心
  - 加高斯偏移（使遮挡不完全对齐关键点）
  - 确保遮挡物覆盖身体区域
- 需要传入 keypoints 和 scores 到增强函数

## 预期结果
- 如果 PA-ROA > random ROA: 证明 pose-guided 放置更好
- 如果 PA-ROA ≈ random ROA: 遮挡位置不重要，只要有遮挡就行
- 如果 PA-ROA < random ROA: pose-guided 放置过于集中，失去了随机性的正则化效果
