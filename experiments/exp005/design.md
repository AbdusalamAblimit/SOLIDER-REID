# exp005: High-Resolution Part Pooling (Stage 2)

## 核心假设

id_part 收敛慢的根本原因是 **空间分辨率不足**：
- Stage 3 特征图仅 12×4 = 48 个空间位置
- 5 个 body part 的 attention 在如此低的分辨率下无法精确区分
- 不同 part 的 pooled features 会高度相似（因为重叠的 attention regions）

## 方案

使用 Stage 2 特征图 (24×8 = 192 positions, 384 channels) 进行 part pooling：
- Global feature 仍从 Stage 3 (12×4, 768ch) GAP 获得
- Part features 从 Stage 2 (24×8, 384ch) pose-guided pooling 获得
- 4× spatial resolution → 更精确的 part attention
- Part feature dim = 384 (vs 之前的 768)

## 架构变化

```
Backbone → Stage 2 output (B, 384, 24, 8) → Part Pooling → 5 × (B, 384)
         → Stage 3 output (B, 768, 12, 4) → GAP → (B, 768) → Global feature

Test: part-only → concat 5 × 384 = 1920-dim feature
```

## 需要修改的模块

1. `PosePartPooling`: 支持 `in_channels=384`（当前是 768）
2. `PoseReIDModel`: 从 `featmaps[-2]` 取 stage 2 特征
3. Config: 新增 `POSE_PART_STAGE` 参数
4. Loss: part triplet 的特征维度从 768 变为 384

## 预期

- id_part 应该更快收敛（更精确的 spatial attention → 更 discriminative 的 part features）
- Part feature 虽然只有 384-dim，但空间信息更丰富可能补偿
- 如果假设正确，最终 mAP 应超过 57.5%

## 风险

- Stage 2 特征可能不够 semantic（还没经过 stage 3 的深层处理）
- 384-dim 可能对 702 个类的分类任务来说不够
