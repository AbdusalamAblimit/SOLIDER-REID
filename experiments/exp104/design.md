# exp104: PACD (Pose-Anchored Contrastive Distillation)

## 核心创新 — 范式级

**训练 backbone 产出本质上遮挡不变的特征**

不加模块，不加分支，不加辅助网络。直接改训练目标：
- 正常 forward → 得到 full feature map (Stage 3)
- 用 pose heatmap 随机 mask 40% 身体部位区域 → masked feature map
- GAP(masked) → partial feature
- Loss: partial feature ≈ full feature (self-distillation)

**零新参数。** 唯一改的是 processor 中的损失函数。

## 为什么之前的辅助损失都失败了，但 PACD 可能不同

| 失败方法 | 失败原因 | PACD 为何不同 |
|---------|---------|-------------|
| SGMKC/SGMT | 在 GCN 分支 mask → 只影响 part branch | PACD 直接 mask backbone feature map → 影响 global feature |
| LSRM/CIPGFR | 需要 same-ID 跨图像恢复 → batch 太小 | PACD 是 same-image self-distillation → 无跨图像依赖 |
| PCQA/PTM | detach target → moving target 问题 | PACD detach 的是 full feature → 非常稳定的 target |
| PAMC | 用 projector + eval mode + masked IMAGE | PACD 直接 mask feature MAP → 更轻量更直接 |

## 对照组
- exp066 (PAA baseline): 61.6%/74.2%
