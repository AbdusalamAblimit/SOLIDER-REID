# 实验 exp061: GCN Keypoint Dropout (GKD)

## 动机
- SGMKC (exp048) 尝试对 GCN 输入做 mask + MSE reconstruction loss → 失败（梯度冲突）
- SPD (exp026) 对 PSG heatmap 做 dropout → 中性
- **核心洞察**: SGMKC 失败是因为额外的 reconstruction loss，不是因为 masking 本身
- **GKD 的想法**: 对 GCN 的 bilinear sampled keypoint features 做随机 dropout（置零），但**不加任何额外 loss**。仅通过 ID + triplet loss 让 GCN 学会从不完整输入中提取特征
- 这是纯数据增强（对 GCN 输入），零参数，零额外 loss

## 创新点 / 核心想法
- **核心假设**: 随机 mask GCN 输入关键点特征，迫使 GCN 利用 skeleton graph 传播从可见关键点推断遮挡关键点
- **与 SGMKC 的关键区别**: 无额外 loss。GCN 在 ID loss 监督下自然学会处理缺失关键点
- **与 SPD 的区别**: SPD 对 PSG heatmap 做 dropout（影响 backbone），GKD 对 GCN 输入特征做 dropout（仅影响 branch）
- **论文叙事**: "在训练时模拟关键点遮挡，让 skeleton GCN 学会遮挡鲁棒的特征传播"

## 技术方案
- 在 SkeletonGCNHead.forward() 中，训练时随机 mask 30% 的关键点特征（置零）
- mask 概率可通过 config 控制
- 仅在训练时生效，测试时所有关键点正常使用
- 实现极简：~5 行代码

## 预期结果
- 如果成功: GCN 学会 skeleton-based feature propagation → equal_concat 提升
- 如果失败: 30% dropout 过强导致 GCN 训练不稳定
- 如果中性: dropout 不提供比 PSG 已有的遮挡鲁棒性更多信息

## 对照组
- Baseline: exp030a (PSG+GCN, 无 GKD)
- 消融变量: 仅增加 GCN 输入 dropout
