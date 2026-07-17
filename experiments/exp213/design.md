# 实验 exp213: Small GCN+PAA+CE+OA-SD + PKC(0.05) + MST(0.1)

## 动机
- PKC=0.05 给 maxsim_hybrid +0.1 mAP (72.3→72.4)
- MST 直接优化 MaxSim distance (exp211 正在跑，结果未知)
- PKC 和 MST 是正交的优化信号: PKC 用 SupCon 推 keypoint features 的全局分布, MST 用 triplet 推 per-pair MaxSim distance
- 假设: 两者结合可能给更大的 MaxSim 增益

## 核心假设
PKC + MST 双重 per-keypoint loss 在低权重下不干扰 CE，同时从两个角度改善 keypoint features。

## 技术方案
- exp206r + PKC weight=0.05 + MST weight=0.1

## 对照组
- exp206r (no extra loss): 72.3/82.9 maxsim
- exp210b (PKC only): 72.4/83.1 maxsim
- exp211 (MST only): TBD
