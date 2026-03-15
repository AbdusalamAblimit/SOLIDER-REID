# 实验 exp064: Probabilistic Keypoint Embeddings (PKE)

## 动机
- exp062 LKU 用标量 uncertainty 加权失败（-1.37% R1）
- 但 LKU 只是改了权重，没有改变特征表示和距离计算
- PKE 是一个更深层的改变：每个 keypoint 特征变成高斯分布 (mu, sigma)
- 距离计算从欧氏距离变为 Mutual Likelihood Score (MLS)
- 参考: Hedged Instance Embeddings (ICLR 2019), uncertainties-for-embeddings (MDPI 2024)

## 创新点 / 核心想法
- **核心假设**: 遮挡导致 keypoint 特征的可靠性不同。将特征建模为分布而非点向量，让 uncertainty 自然涌现
- **与 LKU 的 3 个关键区别**:
  1. 特征表示改变: 点向量 → 高斯 (mu, sigma)
  2. Loss 改变: 标准 triplet → MLS-based triplet
  3. 距离计算改变: Euclidean → MLS distance
- **MLS 距离**: d(p,q) = Σ[(mu_p - mu_q)^2 / (sigma_p^2 + sigma_q^2)] + Σ[log(sigma_p^2 + sigma_q^2)]

## 技术方案
- 修改 SkeletonGCNHead: 输出 (mu, log_sigma) 而非单个特征
- GCN pooling: mu 用 confidence-weighted average，sigma 用 inverse-weighted average
- 训练时: MLS triplet loss（在 keypoint 分布上做 mining）
- 测试时: MLS distance 替代欧氏距离做 matching
- 额外参数: variance head ~768 个参数（仅一个 Linear(768, 768) for log_sigma）

## 预期结果
- 如果成功: uncertainty 与遮挡相关 → 自适应距离 → mAP/R1 提升
- 如果失败: sigma 退化为常数 → 等价于标准距离

## 对照组
- exp030a (标准点向量, 欧氏距离)
