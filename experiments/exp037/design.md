# 实验 exp037: Learnable Keypoint Attention (LKA)

## 编号说明
- 按最初的 visibility 路线命名，`exp037` 原本预留给 visibility-aware graph 方向。
- 当前这个 `exp037` 实际实验内容已经偏离原 visibility 路线，转而用于 `exp035/exp036` 之后的 GCN branch 内部探索。
- 后续引用时应明确这是“关键点聚合权重的可学习化探索”，而不是 visibility 路线的阶段性结论。

## 动机
- exp036 证明 GCN 关键点特征已足够判别（额外 triplet loss 反而 -0.5%）
- 但当前的聚合权重是固定的 ViTPose 检测置信度（score）
- 检测置信度衡量的是"关键点定位准确性"，不是"关键点对身份判别的贡献"
- exp035b 证明权重方案敏感（score*visibility -0.7%），说明聚合权重有优化空间
- 可学习的注意力权重让模型自动发现哪些关键点对 ReID 最重要

## 核心假设
- 用可学习的 MLP 替换固定置信度加权，让模型自适应地关注判别性最强的关键点（如脚/鞋、躯干纹理），降权不够判别的关键点（如遮挡的头部）
- 输入为当前 kp_scores (B, 17)，输出为学习到的 attention weights (B, 17)

## 技术方案

### 修改文件
1. `model/modules/skeleton_gcn.py`: 在 `SkeletonGCNHead` 中添加可学习的 attention MLP
2. `config/defaults.py`: 添加 `POSE_KP_LEARNABLE_ATTN` 开关
3. `model/pose_backbone_model.py`: 传参
4. `model/pose_dual_stream_model.py`: 传参

### 模块设计
```python
self.kp_attention = nn.Sequential(
    nn.Linear(17, 32),
    nn.ReLU(),
    nn.Linear(32, 17),
    nn.Sigmoid()
)
# Zero-init last layer bias for identity start (outputs ~0.5)
nn.init.zeros_(self.kp_attention[2].weight)
nn.init.zeros_(self.kp_attention[2].bias)
```

### 数据流
1. kp_scores: (B, 17) → kp_attention MLP → learned_weights: (B, 17)
2. final_weights = kp_scores * learned_weights（置信度 × 注意力）
3. skeleton_feat = weighted_avg(kp_feats_enhanced, final_weights)

### 关键超参数
- `POSE_KP_LEARNABLE_ATTN`: True/False 开关
- MLP: 17→32→17, sigmoid, ~600 参数
- 乘法组合：保留置信度的遮挡检测功能，叠加可学习的判别性注意力

## 预期结果
- 乐观: +0.5~0.8% mAP（发现了更优的关键点权重分配）
- 现实: +0.1~0.3% mAP（微调效果）
- 悲观: ±0.0%（GCN 消息传递已隐式实现了自适应加权，注意力退化为均匀）

## 对照组
- Baseline: exp035a (PSG + GCN, score weight, equal_concat) = 61.1% mAP / 73.8% R1
- 消融变量: 仅增加 POSE_KP_LEARNABLE_ATTN=True
