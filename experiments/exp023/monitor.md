# exp023: PDS + Stop Gradient 监控日志

## 实验信息
- **配置**: configs/occluded_duke/pose_pds_stopgrad.yml
- **模型**: PoseDualStreamModel (与 exp022 相同，+ stop_gradient)
- **数据集**: Occluded-Duke
- **Backbone**: Swin-Tiny
- **额外参数**: ~8.8M (同 exp022)
- **对照**: exp022 PDS global-only (mAP 57.9%), exp007 PSG-only (mAP 58.3%)

---

### [09:39] 检查点 #1 — 启动确认

**状态**: 🟢正常
**进度**: Epoch 1/120 (~0.8%)

| 指标 | 当前值 | 备注 |
|------|--------|------|
| Total Loss | 16.7 | Epoch 1 Iter 80 |
| ID Global | 6.555 | CE loss, 702 classes |
| ID Part | 6.554 | 与 exp022 初始一致 |
| Tri Global | 10.9 | 快速下降 |
| Tri Part | 11.0 | 略高于 global |
| LR | 4.76e-05 | Warmup 中 |

**观察**: 初始 loss 与 exp022 几乎相同。stop_gradient 不影响初始 forward pass，只影响 backward。Config 确认 POSE_PART_STOP_GRAD=True。
**决策**: 继续

---
