# 实验 exp214: Small GCN+PAA + 3-view (无 OA-SD)

## 动机
- 3-view 训练（多增强视图 CE/triplet 平均）在 Tiny 上有效 (+1.4%)
- 但 Small 的 3-view+CP+OA-SD 出过学习停滞 bug (exp206 3-view)
- 假设: 不用 OA-SD 的纯 3-view 可能在 Small 上也有效
- 3-view 增加训练样本多样性，可能改善 GCN keypoint features

## 核心假设
3-view CE/triplet 训练在 Small GCN+PAA 上提供 +1-2% mAP，不需要 OA-SD。

## 技术方案
- 与 exp206r 相同但: POSE_OA_SD=False, POSE_PARALLEL_AUG=True
- 不用 OA-SD 节省显存，3-view 不需要 CP

## 对照组
- exp206r (1-view + OA-SD): 70.6/82.6 (eq), 72.3/82.9 (maxsim)
- exp210b (1-view + OA-SD + PKC): 70.6/81.8 (eq), 72.4/83.1 (maxsim)
