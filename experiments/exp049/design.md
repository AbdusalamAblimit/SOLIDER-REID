# 实验 exp049: NFC (Neighbor Feature Centralization) 测试时特征增强

## 动机
- Pose2ID (CVPR 2025) 提出 NFC 作为一种简单的 test-time 特征增强方法
- NFC 寻找特征空间中的互近邻，将邻居特征累加到原始特征上
- 这种方法可以平滑噪声并增强同身份样本间的特征相似度
- 需要验证 NFC 在我们的 PSG+GCN 特征上是否有效

## 核心想法
- **不涉及训练**：仅修改测试时的特征处理
- 对已有的 exp030a checkpoint 做测试时 NFC 后处理
- 测试不同 k1/k2 参数组合

## 技术方案
- 修改 `utils/metrics.py`：添加 NFC 函数
- 修改 `config/defaults.py`：添加 NFC 配置选项
- 在特征归一化后、距离计算前应用 NFC
- NFC 应用于 query+gallery 合并的特征集

## 预期结果
- 如果有效，预期 mAP 提升 +0.5~1.5%
- NFC 作为通用后处理方法，与 Re-ranking 类似，不算训练端贡献
- 结果将在 results.md 的 "+NFC 结果" 区域单独报告

## 对照组
- Baseline 对照：exp030a (seed 1234) 各模式的无后处理结果
- 消融变量：NFC k1/k2 参数

## 测试矩阵
1. exp030a global + NFC (k1=2, k2=2)
2. exp030a equal_concat + NFC (k1=2, k2=2)
3. 如果有效，尝试 k1=3,k2=3 和 k1=4,k2=4
