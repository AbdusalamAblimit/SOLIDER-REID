# exp008: PSG + Part Pooling 组合

## 动机

exp007 PSG: mAP 58.3% (+1.7%) — backbone-level pose injection
exp001 Part Pooling (part-only): mAP 57.5% (+0.9%) — post-hoc part pooling

两者利用 pose 信息的方式不同：
- PSG: 修改 backbone 如何提取特征（how features are formed）
- Part Pooling: 利用 backbone 输出进行空间聚合（how features are pooled）

假设：两者互补，组合可能 > 各自的最佳。

## 方案

在 PoseBackboneModel 基础上，加入 Part Pooling branch：

```
Swin Backbone
  → Stage 0-2: unchanged
  → Stage 3: PSG after each block
  → feature_map (B, 768, 12, 4) [pose-aware]
  → GAP → global feat [pose-aware, 768-dim]
  → Part Pooling → 5 part feats [from pose-aware features]
  → Training: global ID + part ID + global triplet + part triplet
  → Test: part_only feature (5×768 = 3840-dim)
```

## 与之前实验的区别

| | exp001 | exp004 | exp007 | exp008 |
|--|---|---|---|---|
| Backbone | standard | standard | PSG | PSG |
| Part Pool | ✓ | ✓ | ✗ | ✓ |
| PFM | ✗ | ✓ | ✗ | ✗ |
| Test feat | part_only | part_only | global | part_only |

## 预期

- exp001 part 特征来自"普通" backbone → mAP 57.5%
- exp008 part 特征来自"PSG-enhanced" backbone → 应该 > 57.5%
- 组合：PSG 让 backbone 特征更 pose-aware → part pooling 的输入更好 → part 特征更 discriminative
- 如果增益叠加：可能达到 58%+ 的 part-only mAP

## 实现

创建一个 PoseBackbonePSGPartModel，继承 PoseBackboneModel，加入 PosePartPooling。
或者更简单：在现有 PoseReIDModel 中加入 PSG 支持。
