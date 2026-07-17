# 实验 exp145: PSG+GCN+PAA+SASA (PAA + 骨架注意力偏置)

## 动机

- exp143 SASA (alpha=0.1) 在 PSG+GCN 基础上完美中性
- PAA 在 PSG+GCN 基础上有效 (+0.9% mAP)
- 测试 SASA 是否能在 PAA 基础上提供额外信息（即 SASA+PAA 是否正交叠加）
- SASA 修改 attention routing，PAA 修改 feature values，理论上正交

## 技术方案

在 exp066 (PAA) config 基础上添加 SASA (alpha=0.1)

## 对照组

- exp066 (PAA only): 61.6% / 74.2%
- exp143 (SASA only): 61.1% / 73.7%

## 预期结果

- 如果正交叠加：~62.0%+ mAP
- 如果不叠加（更可能）：~61.6%（= PAA only）
