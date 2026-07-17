# 实验 exp159: PLBOA + ROA 组合

## 动机
- PLBOA (exp157): +1.6% vs baseline (62.7%)
- ROA (exp058): +0.7% vs baseline (61.8%)
- 测试两者是否正交叠加
- PLBOA 遮挡下半身，ROA 随机位置贴物体——可能互补

## 技术方案
- PSG+GCN+PLBOA+ROA (both p=0.7)
- 每张训练图先 ROA，再 PLBOA（两者独立概率）
