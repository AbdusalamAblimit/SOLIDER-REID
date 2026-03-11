# exp024: PDS+StopGrad without PSG 监控日志

## 实验信息
- **配置**: configs/occluded_duke/pose_pds_stopgrad_nopsg.yml
- **模型**: PoseDualStreamModel (POSE_GLOBAL_PSG=False)
- **数据集**: Occluded-Duke
- **Backbone**: Swin-Tiny
- **额外参数**: ~8.8M (同 exp023，PSG 102K 参数创建但未使用)
- **对照**: exp023 PDS+StopGrad+PSG (mAP 59.5%), exp000 baseline (mAP 56.6%)
- **目的**: 消融实验 — 验证 PSG 在 PDS+StopGrad 架构中的贡献

---

### [12:06] 检查点 #1 — 启动确认

**状态**: 🟢正常
**进度**: Epoch 1/120 (~0.8%)

| 指标 | 当前值 | 备注 |
|------|--------|------|
| Total Loss | 19.1 | Epoch 1 Iter 40 |
| ID Global | 6.554 | CE loss, 702 classes |
| ID Part | 6.555 | 与 exp023 初始一致 |
| Tri Global | 12.4 | 快速下降 |
| Tri Part | 12.7 | 略高于 global |
| LR | 4.76e-05 | Warmup 中 |

**观察**: 训练正常启动。确认 `[PDS] Global branch: Stage 3 (2 blocks) + no PSG (ablation)`。初始 loss 与 exp023 几乎相同（预期，因为初始 forward pass 不受 PSG 影响——PSG 的零初始化意味着初始贡献为零）。PID: 353649
**决策**: 继续

---

### [12:14] 检查点 #2 — Epoch 9

**状态**: 🟢正常
**进度**: Epoch 9/120 (7.5%)

| 指标 | 当前值 | vs exp023 Ep9 | 备注 |
|------|--------|---------------|------|
| ID Global | 5.98 | exp023 ~5.68 (ep10) | 预期：无 PSG 导致 global 收敛更慢 |
| ID Part | 6.44 | exp023 ~6.38 (ep10) | Part 差异不大 |
| Tri Global | 0.63 | — | 正常下降 |
| Tri Part | 0.64 | — | 正常下降 |
| LR | 3.64e-04 | — | Warmup 中 |

**观察**: 训练正常。Global ID loss (5.98) 略高于 exp023 同期 (5.68)，符合预期——无 PSG 的 Global 分支特征质量较差，分类更困难。ETA ~1h45m。
**决策**: 继续，等 epoch 10 评估结果

---
