# exp025: PDS + Delayed StopGrad 监控日志

## 实验信息
- **配置**: configs/occluded_duke/pose_pds_delayed_stopgrad.yml
- **模型**: PoseDualStreamModel (POSE_STOP_GRAD_EPOCHS=30)
- **数据集**: Occluded-Duke
- **Backbone**: Swin-Tiny
- **额外参数**: ~8.8M (同 exp022/023)
- **策略**: 前 30 轮阻断 Part→shared 梯度，31 轮起释放
- **对照**: exp022 (无StopGrad, 57.9%), exp023 (永久StopGrad, 59.5%)
- **重点监控**: Epoch 30→31 过渡期是否有 loss 突变或 mAP 下降

---

### [13:18] 检查点 #1 — 启动确认

**状态**: 🟢正常
**进度**: Epoch 1/120 (~0.8%)

| 指标 | 当前值 | 备注 |
|------|--------|------|
| Total Loss | 19.3 | Epoch 1 Iter 40 |
| ID Global | 6.555 | 与 exp023 初始一致 |
| ID Part | 6.555 | — |
| LR | 4.76e-05 | Warmup 中 |

**确认**: `[PDS] Part stop_grad: delayed (block first 30 epochs, then release)`。前 30 轮行为应与 exp023 完全相同（都是 detach）。PID: 425342
**决策**: 继续

---
