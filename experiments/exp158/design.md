# 实验 exp158: PLBOA + PAA 组合

## 动机
- PLBOA (exp157): +1.6% mAP vs baseline (62.7%)
- PAA (exp066): +0.9% mAP vs baseline (61.6%)
- 测试两者是否正交叠加

## 技术方案
- PSG+GCN+PAA+PLBOA (lower-body, VOC, p=0.7)
- 单变量: 相对 exp066(PAA) 只多了 PLBOA
