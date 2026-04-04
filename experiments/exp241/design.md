# exp241: PPA + GCN 双分支 on Tiny

## 动机
- PPA (exp237): +0.5/-0.4 — end-to-end backbone training
- GCN (exp191): baseline — detached keypoint features for MaxSim
- 组合: PPA 训练更好的 backbone, GCN 在更好的 backbone 上采样 → 双重提升

## 技术方案
同时启用 PPA 和 GCN:
- PPA 在 non-detached features 上做 part assignment → 训练 backbone
- GCN 在 detached features 上做 keypoint sampling → 提供额外 features
- Output: [global, ppa_pooled, ppa_parts..., gcn_skeleton]
- 两者不竞争: PPA 的梯度到 backbone, GCN 的梯度只到 GCN

需要修改 pose_backbone_model.py: 同时调用 PPA 和 GCN，拼接输出。

## 对照组
- exp237 (PPA only): 63.7/75.0 (+0.5/-0.4)
- exp191 (GCN only): 63.2/75.4
