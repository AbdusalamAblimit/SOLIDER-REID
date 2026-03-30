# exp202 Swin-Small + SupCon + Full Architecture 监控

配置: Swin-Small (50M) + SupCon T=0.05 + PSG + PAPE + MS-PSG + STD-PR + PLBOA
对照:
- exp187 (Swin-Tiny + SupCon + 3-view): 64.9/76.6
- Swin-Small PSG-only baseline: ~67.8 mAP (之前数据)

## 检查点

### [17:17] 检查点 #1

**状态**: 正常
**进度**: Epoch 1/120

| 指标 | 当前值 | 备注 |
|------|--------|------|
| supcon | 4.579 | SupCon 正常 |
| id_global | 6.554 | 初始 |
| tri_global | 14.179 | ↓ 快速 |
| GPU Memory | **12.7GB/24GB** | 非常安全！有余量加 3-view 或 OA-SD |
| Speed | — | 待确认 |

### [17:23] 检查点 #2 — 方案调整

**本地**: 3-view + Swin-Small + WITH_CP=True → **8.3GB/24GB** 完美！
**远程**: 1-view + Swin-Small (权重传输中，完成后启动)

### [17:27] 检查点 #3

**本地 exp202b** (3-view + CP): Speed 52.2 samples/s, ETA **8h30m**。
GPU 8.3GB。训练正常但比 Tiny 慢很多（CP 的代价）。
### [17:33] 检查点 #4

**本地 exp202b**: Epoch 3/120. supcon=2.924, id_global=6.543.
### [17:44] 检查点 #5

**方案确定**:
- **本地 exp202b**: 3-view + Small + **WITH_CP=True** (8.4GB, ~52 samples/s, ETA ~8h)
  不开 CP 会 OOM (21.7GB > 24GB)。CP 是必须的。
- **远程 exp202**: 1-view + Small (权重已传完, 已启动)

### [17:49] 检查点 #6

**本地 exp202b**: Epoch 2/120 (3-view + CP)
**远程 exp202**: Epoch 5/120 (1-view, 更快)
### [17:55] 检查点 #7

### [18:00] 检查点 #8

### [18:03] 检查点 #9 — 远程 ep10 🔥

**远程 Swin-Small 1-view ep10**: 43.1/56.4

对比 Tiny 1-view (exp176) 大约 ep10 ~36/48: **+7/+8!!**
Swin-Small 在 ep10 已经接近 Tiny ep20 的水平！

远程 speed=103.2, ETA 4h01m。
本地 3-view+CP, ep4, speed ~52。
### [18:26] Power Norm Test-Time 测试

在 exp187 ep120 (Tiny) 上测试 power normalization:

| alpha | mAP | R1 | delta |
|-------|------|------|------|
| 1.0 (无) | 64.9% | 76.6% | — |
| **0.9** | **64.9%** | **76.9%** | **+0.3 R1** |
| 0.8 | 64.9% | 76.8% | +0.2 R1 |
| 0.5 | 64.5% | 76.3% | -0.4/-0.3 |

alpha=0.9 微提升 R1。将在 Swin-Small 最终 checkpoint 上也测试。
**决策**: 继续
