# 实验 exp210b: GCN+PAA+CE+OA-SD + PKC (weight=0.05)

## 动机
- exp210 (PKC weight=0.5) 灾难性失败: ep10=3.6% mAP
- PKC weight=0.5 的 SupCon 梯度与 CE 在 GCN keypoint features 上严重冲突
- 假设: 极低 weight (0.05) 的 PKC 可以温和地改善 keypoint 特征的 discriminability
- 不破坏 CE 收敛，同时为 MaxSim test-time matching 提供更好的 per-keypoint features

## 核心假设
PKC weight=0.05 足够轻量，不会干扰 CE/triplet 训练，但足以提供额外的 per-keypoint contrastive signal。

## 技术方案
- 与 exp206r 完全相同 + PKC weight=0.05
- PKC 在 GCN 的 17 个 keypoint features 上做 SupCon (visibility threshold=0.3)

## 对照组
- exp206r (fixed OA-SD, no PKC): 70.6/82.6 (equal_concat), 72.3/82.9 (maxsim_hybrid)
- exp210 (PKC weight=0.5): 3.6/5.3 at ep10 (灾难)
