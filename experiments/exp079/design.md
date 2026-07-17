# 实验 exp079: PSG+GCN+ROA (无 PAA)

## 动机
- exp058 ROA 在 PSG+GCN+PAA 上有效 (+1.07% mAP)
- 但 ROA 是否需要 PAA 配合才有效？还是在基础 PSG+GCN 上也有效？
- 这是一个重要的消融：分离 ROA 和 PAA 的独立贡献

## 技术方案
- PSG + GCN (exp030a 基础) + ROA 数据增强
- 不带 PAA
- 对照: exp030a 3-seed mean = 60.73%/72.57%

## 预期
- 如果 ROA 独立有效: mAP > 60.73% (exp030a 3-seed mean)
- 如果 ROA 需要 PAA: ≈ exp030a

## 对照
- exp030a 3-seed = 60.73%/72.57% (baseline)
- exp058 ROA+PAA: 61.8%/72.8%
- exp066 PAA: 61.6%/74.2%
