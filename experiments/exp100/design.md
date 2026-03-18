# 实验 exp100: Pose-FiLM (全阶段姿态条件 FiLM)

## 动机
- PSG 在 Stage 3 做乘法调制 +1.3%，PAA 做加法 +0.9%
- 但 Stages 0-2 完全不知道 pose 信息
- PKP (KPR 式 additive prompting) 效果中性 — 简单加法在 Swin window attention 中传播有限
- **FiLM** 是一种更成熟的条件机制：per-channel affine (scale+shift)
  - 从 VQA 领域引入，首次应用于 person ReID

## 核心想法
用 FiLM (Feature-wise Linear Modulation) 在 backbone 每一层注入 pose 信息：
`output = (1 + γ) × features + β`
其中 γ, β 由 pose heatmap 生成。

与 PSG 的关键区别：
1. **Scale + Shift** vs 仅 Scale（PSG 没有 shift）
2. **每层注入** vs 仅 Stage 3（FiLM 覆盖所有 12 个 Swin blocks）
3. **Channel-wise** vs Spatial（PSG 是空间级，FiLM 是通道级）
4. PSG 和 FiLM 是正交的 — 可以同时使用

## 技术方案
每个 Swin block 后添加 FiLM 层：
1. GAP(heatmap) → pose_feat (B, 17)
2. MLP: 17 → 32 → 2C → gamma (B,C), beta (B,C)
3. 应用: x = (1+gamma)*x + beta
4. Zero-init → 初始恒等映射

参数: 12 blocks × (17×32 + 32×2C) ≈ 290K total

## 对照组
- exp066 (PSG+GCN+PAA): 61.6%/74.2%
