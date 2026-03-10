# 实验 exp018: PCG-only（无 PSG），验证 PCG 独立效果

## 动机
- exp017 PCG+PSG 结果与 PSG-only 持平（mAP 58.0% vs 58.3%）
- 两种可能的解释：
  1. **PSG 已经做了足够的 pose conditioning**，PCG 没有额外信息可加
  2. **PCG 本身就是无效的**，不管有没有 PSG 都不行
- 这个消融实验回答一个关键问题：PCG 单独使用（无 PSG）是否比 baseline 好？
- 如果 PCG-only > baseline，说明 PCG 有独立价值，只是被 PSG 的效果"覆盖"了
- 如果 PCG-only ≈ baseline，说明 PCG 确实无效——GAP 后的 pose 信息不足以调制特征

## 创新点 / 核心想法
- **核心假设**: PCG 在没有 PSG 的情况下，是否能提供有意义的 pose conditioning
- **与 exp017 的区别**: 去掉 PSG，只保留 PCG。`POSE_PSG_STAGES: []` (空列表禁用 PSG)
- **消融逻辑**: 验证 PCG 的独立价值

## 技术方案

### 修改文件
- 仅修改 config：关闭 PSG，只开 PCG
- 需要确保 PoseBackboneModel 在 `POSE_BACKBONE_PSG: False` 时不创建 PSG 模块但仍支持 PCG

### 关键问题
PCG 目前在 `PoseBackboneModel` 中实现，而 `PoseBackboneModel` 只在 `POSE_BACKBONE_PSG: True` 时被实例化。如果关闭 PSG，需要将 PCG 移到基础模型中，或者保持 `POSE_BACKBONE_PSG: True` 但清空 `psg_stage_indices`。

**方案**: 使用 `POSE_BACKBONE_PSG: True`（加载 PoseBackboneModel），但设置 `POSE_PSG_STAGES: []`（空列表，不创建任何 PSG 模块），同时开启 `POSE_CHANNEL_GATE: True`。这样 backbone 照常运行（无 PSG 注入），但 GAP 后有 PCG。

### 数据流
```
Input Image + Pose Heatmaps
  → Swin Backbone Stage 0-3 (完全正常，无 PSG)
  → out_feat: (B, 768, 12, 4)
  → GAP → global_feat: (B, 768)
  → PCG: global_feat * (1 + channel_gate(pose_desc))
  → BN → Classifier
```

### 关键超参数
- `POSE_BACKBONE_PSG: True` (使用 PoseBackboneModel)
- `POSE_PSG_STAGES: []` (空列表，不创建 PSG)
- `POSE_CHANNEL_GATE: True`
- `POSE_PCG_HIDDEN: 64`

## 预期结果
- **如果 PCG-only > baseline**: mAP ~57.0-57.5%，说明 PCG 有独立效果
- **如果 PCG-only ≈ baseline**: mAP ~56.5-57.0%，说明 PCG 无效
- **如果 PCG-only < baseline**: 说明 PCG 有害（不太可能，因为 zero-init）

## 对照组
- Baseline: exp000 (mAP 56.6%, R1 66.5%)
- 对照 exp017 PSG+PCG (mAP 58.0%, R1 67.3%)
- 消融变量: 有无 PSG
