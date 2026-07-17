# 实验 exp028: PDS + Part 分支学习率放大 (Part LR Boost)

## 动机
- exp023 PDS+StopGrad 是目前最佳方法（mAP 59.5%, R1 69.5%）
- 但 Part 分支收敛不充分：exp022 的 Part ID loss 最终为 ~2.0（vs Global ~0.2）
- Part-only mAP 仅 56.7%，concat_scaled (59.1%) 反而低于 global-only (59.5%)
- 假设：Part 分支与 Global 共享优化器和 LR，但 Part 的 Stage 3 是随机初始化的（而 Global 继承预训练权重），因此相同 LR 对 Part 来说偏低
- 如果给 Part 分支一个更大的 LR，Part 特征可能更好，从而使 concat_scaled 超过 global-only

## 创新点 / 核心想法
**核心假设：PDS 的 Part 分支因为随机初始化 + 共享 LR 导致收敛不充分，给 Part 分支独立的 LR 放大因子可以改善 Part 特征质量，使 Global+Part 融合超越 Global-only。**

## 技术方案

### 修改文件
1. **`config/defaults.py`**: 新增 `POSE_PART_LR_FACTOR = 1.0`（默认不改变）
2. **`solver/make_optimizer.py`**: 为 Part 分支参数（`part_stage3`, `part_norm3`, `part_pooling`）设置独立的 LR 放大因子

### 数据流
```
optimizer 构建时:
    - Global/shared params: base_lr
    - Part branch params (part_stage3, part_norm3, part_pooling): base_lr * POSE_PART_LR_FACTOR
    - 注意: Part 分支的 bias 参数会同时受 BIAS_LR_FACTOR(2x) 和 PART_LR_FACTOR(3x) 影响，实际 LR = base_lr * 6
```

### 关键超参数
- `POSE_PART_LR_FACTOR = 3.0`（Part LR 放大 3 倍）
- 选择 3.0 的依据：Part Stage 3 从随机初始化开始，需要更大的 LR 才能在 120 epoch 内充分收敛。3x 是一个保守的放大比例。
- 其余所有参数与 exp023 (PDS+StopGrad) 完全相同

## 预期结果
- **如果假设成立**: Part-only mAP > 57.5%（超过 exp023 的 56.7%），concat_scaled > 59.5%
- **如果中性 ≈ exp023**: Part LR 不是瓶颈
- **如果 < exp023**: 更大的 LR 导致 Part 过拟合或不稳定
- **最可能失败原因**: Part 收敛不充分不是因为 LR 低，而是因为 stop_grad 限制了共享特征的质量

## 对照组
- **直接对照**: exp023 PDS+StopGrad (mAP 59.5%, R1 69.5%)
- **消融变量**: 仅 Part 分支的 LR 放大因子从 1.0 → 3.0
