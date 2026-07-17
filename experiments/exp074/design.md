# 实验 exp074: PAA + PGAM 组合

## 动机
- PAA (exp066): +0.87%/+1.63% vs 3-seed — 训练端最佳创新
- PGAM (exp054): +0.37%/+1.23% vs 3-seed — 唯一正向叠加的 attention 模块
- 两者从未同时使用：
  - PAA 做 additive injection (在 block 之后)
  - PGAM 做 attention masking (在 block 内部)
  - 机制正交，理论上应该可叠加
- exp059 (PGAM+ROA) 显示 PGAM 与 ROA 完全冗余
- 但 PAA ≠ ROA，PAA+PGAM 的交互未知

## 创新点 / 核心想法
- 同时使用三种 pose injection 模式（3-channel pose injection）：
  1. PSG: multiplicative gate (feature magnitude)
  2. PGAM: attention mask (attention routing)
  3. PAA: additive content (feature enrichment)
- 如果有效，形成"Pose-Triple-Injection"的论文叙事

## 技术方案
- **Config-only 改动**: 在 PAA config 上添加 `POSE_ATTN_MASK: True`
- 无代码改动，已有框架支持

## 预期结果
- 正向叠加: mAP +0.3~0.5% over exp066 → 62.0%+ mAP
- 如果冗余（像 exp059）: = exp066

## 对照组
- exp066 PAA: 61.6%/74.2%
- exp054 PGAM (no PAA): 61.1%/73.8%
