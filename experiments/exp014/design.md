# 实验 exp014: PSG + GiLt (Per-Part Triplet Loss)

## 动机
- PSG backbone injection 是当前最佳方法 (+1.7% mAP)
- 但训练信号仍是标准 ID+triplet loss，没有部件级监督
- Phase 1 中 GiLt 在 PCFC 基础上额外 +0.5% mAP，说明 per-part triplet 能提供正交增益
- 所有之前在 PSG 上叠加更多 backbone 模块的尝试（PAB、multi-stage、part pooling）都失败了，因为模块间会互相干扰
- **Loss 级的改进是正交方向，不会改变 backbone 的 forward pass，只提供额外梯度信号**

## 创新点 / 核心想法
- **假设**: 在 PSG 增强的特征图上做 pose-guided part pooling + per-part triplet loss，让 backbone 不仅整体判别，还学习部件级判别性
- 与 exp008 (PSG+Part Pooling) 的区别：exp008 使用 part features 做 **测试**（part-only test），而 exp014 只用 part triplet 做 **训练**，测试仍用 PSG global feature
- 关键区别：GiLt 只在训练时给梯度，不改变测试时的特征提取流程

## 技术方案
- 基于 exp007 PSG backbone model
- 在 Stage 3 输出的 PSG-enhanced feature 上加 pose part pooling（复用 exp001 的 PosePartHead）
- 训练时：全局 ID loss + 全局 triplet + per-part triplet (GiLt)
- 测试时：只用 PSG global feature（不用 part features）
- 新增参数：Part pooling head（~2.6M，但测试时不用）

### 修改文件
1. `model/pose_backbone_model.py` — 加入 part head（训练用）
2. `processor/processor.py` — 加入 per-part triplet loss 计算
3. `configs/occluded_duke/pose_psg_gilt.yml` — 新配置

### 关键超参数
- part_triplet_weight: 1.0（与全局 triplet 同权）
- part_count: 5 (head, torso, upper_legs, lower_legs, feet)
- PSG hidden_dim: 64（与 exp007 相同）
- 测试特征: global only（不用 part）

## 预期结果
- 如果假设成立：mAP 58.5-59.0%（在 PSG 58.3% 基础上 +0.2-0.7%）
- 如果失败：最可能原因是 per-part triplet 的梯度与 PSG gate 的梯度冲突

## 对照组
- Baseline 对照: exp007 (PSG-only, mAP 58.3%)
- 消融变量: 仅增加 GiLt loss，其他不变
