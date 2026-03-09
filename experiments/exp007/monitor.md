# exp007: Pose Backbone PSG — 训练监控

## 实验配置
- **核心改动**: Pose Spatial Gate (PSG) 注入 Stage 3 每个 SwinBlock 之后
- **Config**: `configs/occluded_duke/pose_backbone_psg.yml`
- **Output**: `./log/occluded_duke/exp007_backbone_psg`
- **模型**: PoseBackboneModel (无 part pooling，纯 global feature)
- **额外参数**: 2 × PSG (~102K params)
- **输出特征**: 768-dim global (与 baseline 相同维度)
- **验证**: PSG zero-init 与 baseline 输出完全一致 (diff=0.0)

---
### [19:22] 检查点 #1

**状态**: 🟢正常
**进度**: Epoch 1/120 (~0.8%)

| 指标 | 当前值 | 备注 |
|------|--------|------|
| Total Loss | 15.4-20.2 | 初始阶段，正常 |
| id_global | 6.555 | ln(702)≈6.55, 随机分类器 |
| tri_global | 8.9-13.7 | 下降中 |
| LR | 4.76e-05 | warmup |

**观察**: 只有 global loss，没有 part loss（符合预期）。Loss 初始值与 baseline 一致，PSG zero-init 工作正常。
**决策**: 继续监控
