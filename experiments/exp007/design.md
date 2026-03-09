# exp007: Pose Spatial Gate (PSG) Inside Backbone

## 核心假设

Post-hoc part pooling 上限在 +0.9%（exp001-006）。如果 pose 信息参与 backbone 的特征形成过程（而不是事后利用），global feature 本身就可以 pose-aware。

## 方案

在 Swin-Tiny Stage 3（2 个 SwinBlock）的每个 block 之后注入 Pose Spatial Gate：

```
Stage 3 Block 0: [norm → attention → residual → norm → FFN → residual]
                → PSG[0]: x = x * (1 + gate_0(heatmap))
Stage 3 Block 1: [norm → attention → residual → norm → FFN → residual]
                → PSG[1]: x = x * (1 + gate_1(heatmap))
→ norm → reshape → GAP → global feat (pose-aware)
```

PSG 结构: Conv2d(17→64→768), zero-init, ~51K params per gate, total ~102K extra params.

## 与 PFM (exp004) 的关键区别

| | PFM (exp004) | PSG (exp007) |
|--|---|---|
| 位置 | backbone 输出之后 | backbone 内部 (stage 3 blocks 之间) |
| 影响范围 | 只改变最终 feature map | 影响后续 attention 计算 |
| 是否有 part pooling | 是 (冗余) | 否 (纯 global feature) |
| 输出 | 768 + 5×768 part feats | 768 global feat |
| 累积效应 | 无 | 2次 gating 累积 |

## 预期

1. 如果 PSG 有效，global feat 直接超 baseline（不需要 part branch）
2. 架构更简洁，论文 story 更好
3. 如果 PSG global 能达到 57.5%+ (≈ exp001 part-only)，则证明 backbone injection > post-hoc pooling
4. 后续可以 PSG + part pooling 组合（但要注意冗余性）

## 风险

1. Stage 3 只有 2 个 blocks，PSG 的影响可能不够
2. Zero-init 可能导致 PSG 学习太慢
3. 修改 backbone forward 可能引入 bug（需要验证 backbone 输出一致性）
