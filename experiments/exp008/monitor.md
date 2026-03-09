# exp008: PSG + Part Pooling — 训练监控

## 实验配置
- **核心改动**: PSG backbone injection + Pose Part Pooling 组合
- **Config**: `configs/occluded_duke/pose_psg_part.yml`
- **Output**: `./log/occluded_duke/exp008_psg_part`
- **模型**: PosePSGPartModel (PSG in Stage 3 + 5-part pooling)
- **Test feat**: part_only (5×768 = 3840-dim)
- **GPU**: 7.7GB

---
### [21:29] 检查点 #1

**状态**: 🟢正常
**进度**: Epoch 1/120

| 指标 | 当前值 |
|------|--------|
| id_global | 6.555 |
| id_part | 6.555 |
| tri_global | 10-14 |
| tri_part | 10-14 |

**观察**: 初始阶段正常。注意 tri_part 与 tri_global 几乎相同（exp001 中 tri_part 初始值约 2.5）——这是因为 PSG 初始化为 identity，所以 part pooling 输入与 baseline 相同。
**决策**: 继续监控
