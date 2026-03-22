# 实验 exp154: PAA + MaxSim Additive Triplet

## 动机
- exp153 (additive MaxSim) 到 ep90 与 exp030a 基本持平（+0.4 mAP），没有崩溃
- PAA (exp066) 是已确认有效的训练组件 (+0.9% mAP)
- 测试 MaxSim additive 是否与 PAA 正交叠加

## 技术方案
- 基线: exp066 (PSG+GCN+PAA)
- 改动: 增加 MaxSim additive triplet (weight=0.25, tau=0.05)
- 单变量: 相对 exp066 只多了 MaxSim additive triplet

## 对照组
- exp066 PAA only (61.6%/74.2% equal_concat)
- exp153 MaxSim additive without PAA (进行中)

## 预期结果
- 如果 MaxSim+PAA > PAA: 两者正交，MaxSim 是有效的辅助训练信号
- 如果持平: MaxSim additive 权重太弱，被 PAA 覆盖
