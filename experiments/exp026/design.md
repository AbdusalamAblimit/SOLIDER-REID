# 实验 exp026: Stochastic Pose Dropout (SPD)

## 动机
- 25 个实验表明，PSG (58.3%) 是我们最强的单模块贡献，但所有在 PSG 基础上添加新模块的尝试都因梯度干扰而失败
- PDS 实验 (exp022-025) 的 +2.6% 提升高度可疑，极可能是训练随机性差异（待多 seed 验证）
- 关键问题：PSG 的 58.3% 是否是上限？还是 backbone 对 pose 信号过度依赖，导致内在判别力退化？
- 灵感来源：Dropout/Cutout/Stochastic Depth 等随机正则化方法在深度学习中被广泛验证有效

## 创新点 / 核心想法
**核心假设：PSG 让 backbone 过度依赖外部 pose 空间先验，削弱了自身学习的判别特征。通过训练时随机丢弃 pose 信号（但测试时始终使用），可以迫使 backbone 在有无 pose 两种模式下都学会判别，同时仍能在测试时受益于 pose 引导。**

与 baseline / 前序实验相比：
- 基于 exp007 (PSG) 架构，不使用 PDS 双分支
- 唯一改动：训练时以概率 p 将 scene_heatmaps 置零（PSG 的 zero-init 确保此时 gate=0，即 x*(1+0)=x，退化为恒等）
- 测试时始终使用 pose 信号

## 技术方案

### 修改文件
1. **`config/defaults.py`**: 新增 `POSE_DROPOUT_P = 0.0`（默认不启用，向后兼容）
2. **`model/pose_backbone_model.py`**: 在 `forward()` 中，训练时以 `self.pose_dropout_p` 概率将 `scene_heatmaps` 置零

### 数据流
```
输入 → Swin Stage 0-2 → Stage 3 (with PSG)
                              ↓
                  训练时: p 概率 scene_heatmaps → zeros
                         (1-p) 概率保持原值
                  测试时: 始终使用原值
                              ↓
                    GAP → BN → Classifier
```

### 关键代码变更（约 5 行）
```python
# In PoseBackboneModel.forward():
if self.training and scene_heatmaps is not None and self.pose_dropout_p > 0:
    keep_mask = (torch.rand(scene_heatmaps.shape[0], 1, 1, 1,
                            device=scene_heatmaps.device) >= self.pose_dropout_p)
    scene_heatmaps = scene_heatmaps * keep_mask.float()
```

### 注意：zero-init 与训练过程中的行为差异
PSG 的 zero-init 确保**训练开始时**，dropout 的样本退化为恒等操作 `x*(1+0)=x`。但随着训练进行，PSG 编码器参数不再为零，此时被 dropout 的样本会收到 `x*(1+encoder(sigmoid(0)))` 的固定调制（sigmoid(0)=0.5 是常数输入）。这意味着 dropout 样本获得的是一个"学习到的平均空间调制"而非严格恒等。这可能实际上是有益的——backbone 被迫在"完整 pose 引导"和"固定平均引导"两种模式下都学好。

### 关键超参数
- `POSE_DROPOUT_P = 0.3`（初始值，参考标准 dropout 范围 0.1-0.5）
- 其余所有参数与 exp007 (PSG) 完全相同

### 为什么选择 per-sample 而非 per-token 或 per-channel
- Per-sample 最简单且语义清晰：整张图要么用 pose 要么不用
- Per-channel (per-keypoint) 会丢弃部分身体结构信息，不太直观
- 后续可以尝试 per-channel dropout 作为变体（exp026b）

## 预期结果
- **如果假设成立**：mAP > 58.3%（超过 PSG baseline），说明 pose dropout 正则化有效
- **如果 mAP ≈ 58.3%**：说明 backbone 没有过度依赖 pose，但 dropout 也不损害性能（中性结果）
- **如果 mAP < 58.3%**：说明 pose 信号始终有用，dropout 只是移除了有用信息（负面结果，但有信息价值）
- **最可能的失败原因**：pose 热图在大多数样本上确实提供了准确且有用的空间先验，dropout 只是降噪作用有限

## 对照组
- **Baseline 对照**: exp000 (mAP 56.6%)
- **直接对照**: exp007 PSG (mAP 58.3%, R1 67.9%)
- **消融变量**: 本实验相对于 exp007 只增加了一个超参数 `POSE_DROPOUT_P=0.3`

## 论文定位
- 如果有效，可以写成"Stochastic Pose Regularization"，与 PSG 形成正交贡献
- 消融表格可展示不同 dropout rate 的效果
- 论文 story: "PSG teaches the backbone WHERE to attend, SPD prevents over-reliance and maintains robust intrinsic representations"
