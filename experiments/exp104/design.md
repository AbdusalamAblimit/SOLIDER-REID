# 实验 exp104: PACD (Pose-Anchored Contrastive Distillation)

## 动机
- 为什么要做这个实验？
  - 102个实验证明：所有辅助模块/损失在15K数据集上失败
  - SGCFR (+2.6%) 成功的原因：backbone 特征在遮挡时退化
  - **核心问题重定义**：不是"如何恢复遮挡特征"，而是"如何训练出本质上遮挡不变的特征"
- 基于哪些前序实验的发现？
  - exp101 SGMT (-0.6%): 在 GCN 分支 mask → 只影响 part，不影响 global
  - PAMC (失败): 用 projector + eval mode + masked IMAGE → 太复杂
  - SGCFR (+2.6%): 证明 backbone 在遮挡时确实退化

## 创新点 / 核心想法
- 本实验验证的核心假设：
  **通过 pose-guided feature map masking + self-distillation，可以训练 backbone 产出遮挡不变的全局特征，无需任何新参数**
- 与 baseline / 前序实验相比，改了什么？
  - 仅在 processor 中新增 PACD loss（~40行代码）
  - 零新参数、零新模块

## 技术方案
- 修改了哪些文件？
  - `processor/processor.py`: 新增 PACD loss 计算（mask feature map → re-pool → MSE with full feat）
  - `config/defaults.py`: POSE_PACD 配置项
  - `configs/occluded_duke/pose_psg_gcn_paa_pacd.yml`: 实验配置
- 数据流：
  1. 正常 forward → Stage 3 feature map (B, 768, 12, 4) + global feat (B, 768)
  2. 从 pose heatmap 选 40% 关键点 → 生成空间 mask
  3. feature map * (1 - mask) → GAP → feat_partial
  4. L_pacd = MSE(feat_partial, feat_full.detach())
- 关键超参数：
  - PACD_WEIGHT = 0.3
  - PACD_MASK_RATIO = 0.4 (mask 7/17 关键点)
  - PACD_WARMUP = 10 epochs

## 预期结果
- 如果假设成立：mAP +0.5~1.5%（backbone 学会遮挡不变表示）
- 如果失败：最可能原因是 feature map 12×4 分辨率太低，mask 过于粗糙

## 对照组
- Baseline 对照：exp066 (PSG+GCN+PAA) 61.6%/74.2%
- 消融变量：仅新增 PACD loss（零新参数）
