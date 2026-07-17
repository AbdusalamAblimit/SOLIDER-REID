# 实验 exp073: Multi-Stage PSG+PAA (Stage 2+3)

## 动机
- PSG+PAA 目前只在 Stage 3 (最后 2 blocks, 768 dims)
- Stage 2 有 6 blocks (384 dims), 是语义最丰富的中间阶段
- exp009 (multi-stage PSG alone) 与单 stage PSG 持平
- 但 exp009 没有 PAA。PSG+PAA 的双通道注入可能在 Stage 2 也有价值
- Stage 2 的空间分辨率更高 (24×8 vs 12×4), 热图可以更精确地映射

## 创新点 / 核心想法
- 将 PSG+PAA 从 Stage 3 扩展到 Stage 2+3
- Stage 2: 6 blocks × (PSG + PAA), 384 dims
- Stage 3: 2 blocks × (PSG + PAA), 768 dims
- 更多 blocks 意味着 pose 信号更深入地影响特征形成

## 技术方案
- **修改**: 仅改 config，使用 `POSE_PSG_STAGES: [-2, -1]` (Stage 2+3)
- **无代码改动**: 现有框架已支持多 stage PSG/PAA
- **参数增加**: Stage 2 新增 6 × (PSG 384-dim + PAA 384-dim) ≈ extra ~200K

## 预期结果
- 如果多 stage 注入更有效: mAP +0.3~0.5% over exp066
- 如果中性: 说明 Stage 3 注入已经足够
- exp009 的先例暗示可能中性，但 PAA 的加入可能改变这一点

## 对照组
- exp066 PAA (Stage 3 only): 61.6%/74.2%
- exp009 Multi-stage PSG (no PAA): 58.3%/67.2% (= exp007)
