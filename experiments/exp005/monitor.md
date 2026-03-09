# exp005: Stage 2 High-Resolution Part Pooling — 训练监控

## 实验配置
- **核心改动**: Part pooling 使用 Stage 2 特征图 (24×8, 384ch) 而非 Stage 3 (12×4, 768ch)
- **Config**: `configs/occluded_duke/pose_stage2_parts.yml`
- **Output**: `./log/occluded_duke/exp005_stage2_parts`
- **Heatmap norm**: sigmoid
- **Test feat**: part_only
- **POSE_PART_STAGE**: -2 (stage 2)
- **Part feature dim**: 384 (vs 768 in exp001-004)
- **Spatial resolution**: 24×8=192 positions (vs 12×4=48)

---
### [18:12] 检查点 #1

**状态**: 🟢正常
**进度**: Epoch 1/120 (~0.8%)

| 指标 | 当前值 | 备注 |
|------|--------|------|
| Total Loss | 12.5-14.8 | 初始阶段，正常 |
| id_global | 6.555 | 刚开始，接近 ln(702)=6.55 |
| id_part | 6.554 | 与 global 几乎一致（随机分类器阶段） |
| tri_global | 9.8-13.8 | 快速下降中 |
| tri_part | 2.2-2.7 | 比 exp001 初始值更低？需观察 |
| LR | 4.76e-05 | warmup 阶段 |

**观察**: 训练正常启动。初始 id_part 与 id_global 一致（6.554），因为都是随机初始化的分类器。tri_part 初始值比 exp001 低，可能是 384ch 的特征导致 triplet margin 更容易满足。
**决策**: 继续监控
