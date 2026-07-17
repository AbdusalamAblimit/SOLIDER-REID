# 实验 exp007a: PSG + 0.5x Global Loss Scale（验证 loss weighting 效应）

## 动机
- codex 分析发现：exp023 (PDS+StopGrad) 的增益主因不是 Part 分支教了 Global，而是 loss 聚合中 global loss 被自动乘了 0.5
- 多 seed 验证：exp023 稳定高于 exp007 约 +1.4% mAP（三 seed 平均），这更像系统性优化差异而非随机误差
- **核心问题**：如果在 exp007（PSG-only）的基础上，显式将 global loss 乘 0.5，是否能复现 exp023 的增益？

## 创新点 / 核心想法
- **核心假设**：PDS+StopGrad 的增益来自 loss weighting 的隐式正则化效果，而非双流架构或 Part 分支
- 这是一个**关键消融实验**：如果 exp007a ≈ exp023，则证实 loss weighting 是主因；如果 exp007a << exp023，则说明 PDS 架构本身也有贡献

## 技术方案
- 基于 exp007（PoseBackboneModel + PSG）
- 在 loss 函数中添加 `GLOBAL_LOSS_SCALE` 参数，将 global ID loss 和 global triplet loss 都乘以此系数
- `GLOBAL_LOSS_SCALE = 0.5` 精确模拟 PDS 中 `w_g = 1/(1+1) = 0.5` 的效果
- **无任何架构变化**：不添加 Part 分支、不修改 backbone

### 修改文件
1. `config/defaults.py`: 新增 `_C.MODEL.GLOBAL_LOSS_SCALE = 1.0`
2. `loss/make_loss.py`: 在非 list score/feat 路径中应用 scale
3. `configs/occluded_duke/pose_psg_half_loss.yml`: 新配置

### 关键超参数
- `GLOBAL_LOSS_SCALE = 0.5`（与 PDS 的隐式 0.5 完全对应）
- 其余所有超参数与 exp007 完全相同

## 预期结果
- **如果假设成立** (loss weighting 是主因): exp007a ≈ 59-59.5% mAP（接近 exp023 的 59.5%）
- **如果假设不成立** (PDS 架构有独立贡献): exp007a ≈ 58-58.5% mAP（高于 exp007 但低于 exp023）
- **如果 loss weighting 有害**: exp007a < 58% mAP

## 对照组
- exp007 (PSG-only, 1.0x loss): mAP 58.3%, R1 67.9%
- exp023 (PDS+StopGrad, 隐式 0.5x): mAP 59.5%, R1 68.5%
- 消融变量：仅 GLOBAL_LOSS_SCALE (1.0 → 0.5)

## 论文意义
- 如果确认 loss weighting 是主因，这将重新定义我们对 PDS 架构的理解
- 可以写成 "the improvement attributed to dual-stream architecture is primarily a loss scaling effect"
- 这个发现本身就有论文价值（负面结果揭示了一个被忽视的 confounding factor）
