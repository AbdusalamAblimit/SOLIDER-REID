# exp001_baseline 监控日志

## 实验信息
- **Config**: `configs/occluded_duke/exp001_baseline.yml`
- **Backbone**: Swin-Tiny (WITH_CP=True)
- **数据集**: Occluded-Duke (702 IDs, 15618 train, 2210 query, 17661 gallery)
- **GPU**: RTX 4070 Laptop (8GB)
- **Loss**: Soft Triplet + ID (no label smoothing)
- **Optimizer**: SGD, LR=0.0008, cosine warmup 20 epochs
- **Epochs**: 120
- **Commit**: exp/phase2 branch

---
### [11:25:30] 检查点 #1

**状态**: 🟢正常
**进度**: Epoch 1/120 (0.8%)

| 指标 | 当前值 | 变化趋势 |
|------|--------|----------|
| Total Loss | 19.42 | — 初始值 |
| Acc | 0.001 | — 初始值 |
| LR | 4.76e-05 | ↑ warmup |
| GPU Mem | 6514/8188 MiB | — |
| GPU Util | 93% | — |

**观察**: 训练正常启动，显存 6.5G，余量约 1.6G
**决策**: 继续
