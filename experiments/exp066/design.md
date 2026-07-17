# 实验 exp066: Pose Additive Adapter (PAA)

## 动机
- PSG 在 Stage 3 block 之后做乘性 gate（+1.33% mAP）
- 所有在 PSG 之上/之后的修改都失败了
- **新方向**: 在 Stage 3 block **内部** 插入轻量 adapter，由 pose 驱动
- 参考: LoRA/Adapter Tuning 在 PEFT (Parameter-Efficient Fine-Tuning) 中的成功
- 与 PSG 的区别: PSG 在 block 输出上做空间 gate；PCA 在 block FFN 中做通道 residual

## 创新点 / 核心想法
- **核心假设**: 在 Swin block 的 FFN 层中间插入一个小型 pose-conditioned bottleneck adapter，让 pose 信息在特征形成的更内部位置起作用
- **机制**: FFN 通常是 Linear→GELU→Linear。在中间加一个 down_proj(pose_feat) → up_proj 的 residual
- **创新**: Pose-conditioned adapter 是 ReID 中未被探索的方向

## 技术方案
- 在 PSG gate 之后，加一个加法 adapter:
  - PSG: x = x * (1 + gate(heatmap))  [乘法, 已有]
  - PAA: x = x + adapter(heatmap)     [加法, 新增]
  - adapter: Conv2d(17→64→768), zero-init output layer
  - 与 ControlNet 加法注入类似
- 与 PSG 同时工作，但提供不同的信号：
  - PSG 调制特征幅值（哪里重要）
  - PAA 添加 pose-derived 特征（什么是身体）

## 预期结果
- 如果成功: PSG×gate + PAA+adapter 互补 → 超越单 PSG
- 如果失败: 加法信号与 PSG 乘法信号冗余
