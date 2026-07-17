# 实验 exp226: 2-Stage Keypoint Fusion with Zero-Init Projection (Tiny)

## 动机
- exp224 (same fusion, random-init proj): 60.7/73.0 (-2.5/-2.4 vs OA-SD)
- 审查指出 kamp_s2_proj 的 Kaiming init 导致 50% 随机噪声
- **修复**: 零初始化 kamp_s2_proj，初始行为 = 纯 Stage 3 (identity start)

## 核心假设
零初始化让模型从"只用 Stage 3"开始，逐渐学习引入 Stage 2 信息，
避免了随机投影噪声对训练的干扰。

## 技术方案
- 与 exp224 完全相同，仅 kamp_s2_proj 增加零初始化
- kamp_scale_attn 已有零初始化 (softmax([0,0]) = [0.5, 0.5])
- 但 kamp_s2_proj 零初始化后，0.5 * zeros = 0 → 初始只用 Stage 3

## 对照组
- exp224 (random-init): 60.7/73.0
- exp191 OA-SD-only: 63.2/75.4
