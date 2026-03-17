# 实验 exp091: Training-Time Skeleton Feature Recovery (TTSFR)

## 动机
- exp090 SGCFR 证明: 跨图片 keypoint feature recovery 在 test-time 有 +2.5% mAP
- 但 SGCFR 是 test-time only，不改训练
- **核心洞察**: 如果我们在训练时就做 skeleton feature recovery，模型可以学到更好的 keypoint 表示

## 创新点
不是加辅助 loss，而是**改变参与主 loss 的特征本身**。

在每个 batch 中：
1. 对同 ID 的正样本对 (A, B)
2. 用 B 的可见 kp 特征替换 A 的遮挡 kp 特征
3. 用恢复后的 kp 特征重新计算 GCN branch feature
4. 恢复后的 feature 也参与 ID + Triplet loss

这和 CIPGFR 的本质区别：
- CIPGFR: 加一个 MSE loss 让遮挡 kp 接近可见 kp → 辅助 loss，失败了
- TTSFR: 直接用恢复后的 kp 特征替换 → 改变主特征，不加新 loss

## 技术方案
修改 GCN branch 的 forward pass：
```
在 SkeletonGCNHead.forward() 中:
  1. 正常计算 kp_feats, kp_weights
  2. 在 batch 内找同 ID 的正样本对
  3. 对遮挡关键点，用正样本的可见特征替换
  4. 用恢复后的 kp_feats 通过 GCN → branch feature
  5. 返回恢复后的 feature 参与 ID + Triplet 主 loss
```

训练时: 恢复特征参与主 loss → 模型学到遮挡鲁棒的表示
测试时: 用 SGCFR 做 gallery-based recovery

## 对照
- exp066 PAA = 61.6%/74.2%
- exp090 SGCFR test-time = 64.1%/75.8%
