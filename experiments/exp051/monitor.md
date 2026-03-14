# exp051 PAML 训练监控日志

## 实验信息
- **方法**: PSG + Skeleton GCN + PAML (Pose-Aware Metric Learning)
- **配置**: `configs/occluded_duke/pose_psg_gcn_paml.yml`
- **输出**: `log/occluded_duke/exp051_paml/`
- **对照**: exp030a (PSG+GCN, equal_concat) 3-seed mean = 60.73% mAP / 72.57% R1
- **核心改动**: Part triplet loss 使用逐关键点 confidence 加权 pairwise 距离（替代聚合 skeleton feature 距离）
- **启动时间**: 2026-03-14 03:01

---

### [03:02] 检查点 #1

**状态**: 🟢正常
**进度**: Epoch 1/120 (~0.8%)

| 指标 | 当前值 | 变化趋势 |
|------|--------|----------|
| Total Loss | 20.09 | ↓ 下降中 |
| ID Global | 6.555 | — |
| ID Part | 6.673 | — |
| Tri Global | 12.546 | ↓ |
| Tri Part (PAML) | 14.406 | ↓ |
| Acc | 0.1% | 初始 |
| LR | 4.76e-05 | warmup |

**观察**: 训练正常启动。PAML tri_part (14.4) 明显高于 exp030a 的标准 tri_part (~6-7)，因为逐关键点距离初始时更大。这是预期行为——模型需要时间优化逐关键点距离。
**决策**: 继续，2 分钟后检查

---

### [03:05] 检查点 #2

**状态**: 🟢正常
**进度**: Epoch 4/120 (~3%)

| 指标 | 当前值 | 变化趋势 |
|------|--------|----------|
| Total Loss | 9.97 | ↓ 稳定下降 |
| ID Global | 6.514 | ↓ 极缓 |
| ID Part | 6.048 | ↓ 极缓 |
| Tri Global | 1.738 | ↓↓ 快速下降 |
| Tri Part (PAML) | 5.637 | ↓↓ 从 16.2→5.6 |
| Acc | 4.9% | ↑ 开始学习 |
| LR | 1.66e-04 | warmup |

**观察**: PAML tri_part 从 16.2 快速下降到 5.6，表明模型正在优化逐关键点距离。tri_global 更快下降到 1.7（标准行为）。训练速度正常（~241 samples/s, ETA 1h49m）。
**决策**: 继续，3 分钟后检查
