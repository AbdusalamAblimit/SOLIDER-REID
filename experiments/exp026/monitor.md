# exp026: PSG + Stochastic Pose Dropout (SPD) 监控日志

## 实验信息
- **配置**: configs/occluded_duke/pose_psg_spd.yml
- **模型**: PoseBackboneModel (PSG + SPD, POSE_DROPOUT_P=0.3)
- **数据集**: Occluded-Duke
- **Backbone**: Swin-Tiny
- **对照**: exp007 PSG (mAP 58.3%, R1 67.9%)
- **目的**: 验证 Stochastic Pose Dropout 能否通过正则化提升 PSG 性能

---

### [18:13] 检查点 #1 — 启动确认

**状态**: 🟢正常
**进度**: Epoch 1/120 (~0.8%)

| 指标 | 当前值 | 备注 |
|------|--------|------|
| Total Loss | 19.1 | Epoch 1 Iter 40 |
| ID Global | 6.555 | CE loss, 702 classes |
| Tri Global | 12.553 | 正常初始值 |
| LR | 4.76e-05 | Warmup 中 |

**观察**: 训练正常启动。确认 `[PSG] Stochastic Pose Dropout enabled: p=0.3`。初始 loss 与 exp007 相同（预期，因为 PSG 的 zero-init 使得前几步 dropout 与否几乎无影响）。PID: 712968
**决策**: 继续
