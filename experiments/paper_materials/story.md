# 论文故事线（持续更新）

## 暂定标题
VPReID: Visibility-aware Pose-guided Person Re-Identification with Adaptive Part Feature Learning

## Motivation（为什么做这个）
- **现有问题**: 遮挡行人重识别中，被遮挡的身体部位产生噪声特征，严重干扰检索
- **现有方法的不足**:
  1. 对遮挡部位只采取"回避"策略（降权或丢弃），浪费了遮挡区域的结构信息
  2. 姿态信息仅用于部件划分和距离计算，未深入影响特征学习过程
  3. SOLIDER预训练的语义-外观解耦能力未与姿态信息协同利用
- **我们的洞察**: ViTPose的VisPredictHead提供了显式的"是否可见"预测（不同于热图置信度），这个信号可以作为条件输入，动态调节每个部件的特征学习策略

## 核心贡献（预计 3 点）
1. 提出 VPReID 框架，利用 ViTPose 的显式可见性预测作为条件信号，实现可见性感知的部件特征学习
2. 设计零参数的 PosePartHead 模块，通过姿态热图软注意力实现语义一致的部件划分，无需额外标注
3. 提出 Visibility-Guided Feature Centralization (VGFC)，在推理时自适应增强部件级特征

## 方法概述
（随着实验推进逐步完善）

VPReID = Clean Swin-Tiny (SOLIDER pretrained) + Frozen ViTPose + PosePartHead + Multi-granularity Training

## 实验证据链
（每个关键实验结果如何支撑我们的 story）
- [ ] exp001: Baseline性能确认
- [ ] exp002: VPReID v1 验证姿态引导有效性
- [ ] exp003-004: 消融实验证明各组件必要性
- [ ] exp007-009: 推理端优化的贡献

## 与 SOTA 对比的 narrative
（待实验结果填充）

## 待补充的实验 / 待解决的问题
- [ ] 运行 dry_run_vpreid.py 验证代码正确性
- [ ] exp001/exp002 获取初始结果
- [ ] 基于结果决定是否追加创新点 A (VCPFL) 或 B (PASM)
- [ ] 可视化分析: attention map, t-SNE, 检索结果
