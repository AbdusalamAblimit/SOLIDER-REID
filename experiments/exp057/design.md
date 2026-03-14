# 实验 exp057: Keypoint Dissimilar Loss (KDL)

## 动机
- 诊断发现 GCN 的 17 个关键点特征存在严重坍缩：**平均余弦相似度 0.694，最高 0.996**
- 不同关键点（如头 vs 脚）的特征应该编码不同的身体部位信息，但当前特征高度相似
- ProFD (ACM MM 2024) 使用 Dissimilar Loss 解决了相同问题（part 特征坍缩）
- KDL 是正则化而非新训练信号——与之前失败的 5 种 auxiliary loss 本质不同

## 创新点 / 核心想法
- **核心假设**: GCN 关键点特征的高相似度限制了 branch 的判别力；通过 Dissimilar Loss 推动特征分散化，可以提升 fusion 后的整体性能
- **与之前失败方向的区别**: CSGT/SGMKC/PAMC/PAML 都是加新的训练目标（新信号），KDL 是对已有特征的正则化（约束）。前者增加梯度冲突，后者减少特征冗余

## 技术方案
- 在 loss 计算中加入 Keypoint Dissimilar Loss:
  ```
  KDL = -mean(pairwise_cosine_distance(kp_feats))
      = mean(cosine_similarity(kp_i, kp_j)) for all i≠j
  ```
- 即最小化不同关键点特征之间的余弦相似度
- 参考 ProFD 的实现（~20 行代码）
- 权重: 0.1（初始值，较小以避免过度正则化）

## 预期结果
- 如果成功: 关键点特征分散化 → equal_concat 模式下 mAP/R1 提升
- 如果失败: 正则化过强导致训练不稳定
- 如果中性: 特征坍缩不是性能瓶颈（坍缩的特征已经包含足够信息）

## 对照组
- Baseline: exp030a (PSG+GCN, 无 KDL)
- 消融变量: 仅增加 KDL loss
