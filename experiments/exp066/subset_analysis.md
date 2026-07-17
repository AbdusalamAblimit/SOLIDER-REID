# exp066 PAA Subset Analysis

## 方法
将 query set 按 pose_data 中的 num_persons 分为 single_person (n=1, 1120 queries) 和 multi_person (n>=2, 1090 queries)。
对 exp030a (baseline) 和 exp066 (PAA) 的 seed1234 checkpoint 分别计算子集 mAP/R1。

## 核心发现

| Subset | Baseline mAP | PAA mAP | Δ mAP | Baseline R1 | PAA R1 | Δ R1 |
|--------|-------------|---------|-------|------------|--------|------|
| all | 60.56% | 61.63% | **+1.07%** | 74.03% | 74.21% | +0.18% |
| single_person (n=1) | 59.17% | 59.63% | +0.47% | 71.88% | 70.27% | **-1.61%** |
| **multi_person (n>=2)** | 61.99% | **63.68%** | **+1.69%** | 76.24% | **78.26%** | **+2.02%** |
| n=2 | 63.61% | 65.44% | +1.83% | 78.17% | 80.99% | +2.82% |
| n=3 | 62.24% | 63.29% | +1.05% | 74.75% | 75.74% | +0.98% |
| n>=4 | 57.41% | 59.62% | +2.21% | 73.27% | 74.65% | +1.38% |

### Target Score 分析

| Subset | Baseline mAP | PAA mAP | Δ mAP | Δ R1 |
|--------|-------------|---------|-------|------|
| target_score LOW | 61.04% | 62.12% | +1.08% | +0.95% |
| target_score MID | 59.86% | 61.45% | +1.59% | +0.68% |
| target_score HIGH | 60.77% | 61.32% | +0.55% | **-1.09%** |

## 关键结论

1. **PAA 是 multi-person occlusion specialist**：增益几乎全部来自多人图 (+1.69%/+2.02%)
2. **单人图 R1 退化 -1.61%**：PAA 的 pose adapter 在无遮挡时产生干扰
3. **target_score HIGH 的 R1 也退化 -1.09%**：高置信度样本（通常是清晰无遮挡图）不需要 PAA
4. **n>=4 场景 mAP 增益最大 (+2.21%)**：越复杂的多人场景，PAA 越有效

## 对论文的意义
- PAA 不是通用 feature enhancer，而是 multi-person occlusion-specific 改进
- 可以支撑 "pose adapter addresses non-target pedestrian interference" 的叙事
- 消融表格应分 single/multi 报告

## 对 TDPC/ST-PAA 的解释
- TDPC 失败因为 74% 训练数据是单人图，TDDA 在这些图上只增加噪声
- 未来方向应考虑：只在多人图上激活额外模块（或加权更高）

## 注意
- 上述对比基于 exp030a (concat_scaled) vs exp066 (equal_concat) 不同的 test feat 模式
- 需要在相同 test feat 模式下做更严格的对照
