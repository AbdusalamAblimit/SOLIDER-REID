# exp246: LGPA-D + GCN 双分支 (语义+结构互补)

## 动机
exp244 LGPA-D (+2.1) 证明 CLIP 语义 part assignment 有效。
exp191 GCN 提供 skeleton graph 结构特征。
两者理论互补: LGPA 做语义级 part pooling, GCN 做骨架级 keypoint 特征。
类似 PPA+GCN (exp241), 但用 LGPA-D 替换 PPA。

## 核心假设
LGPA-D 的语义 part features 与 GCN 的 skeleton keypoint features 正交互补，
双分支 concat 应超过单分支。

## 技术方案
- LGPA-D (detached) + GCN (detached) 双分支
- 已有代码支持: pose_backbone_model.py LGPA+GCN dual branch path
- Test: equal_concat = global + LGPA parts + GCN pooled
- 其他: OA-SD + PLBOA(0.7) + PSG

## 代码修改
仅 config: 同时开启 POSE_LGPA=True + POSE_SKELETON_GCN=True + POSE_LGPA_DETACH=True

## 对照组
- exp244 (LGPA-D only): 65.3/75.7 (+2.1/+0.3)
- exp191 (GCN only): 63.2/75.4
