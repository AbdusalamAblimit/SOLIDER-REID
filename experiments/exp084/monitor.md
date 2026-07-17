# exp084 CIPGFR 训练监控

## 实验信息
- **方法**: PSG + GCN + PAA + CIPGFR (Cross-Instance Pose-Guided Feature Recovery)
- **配置**: `configs/occluded_duke/pose_psg_gcn_paa_cipgfr.yml`
- **对照**: exp066 PAA = 61.6%/74.2%
- **核心**: 跨图片 keypoint feature recovery loss (warmup 20ep)
- **启动**: 2026-03-17 00:25, PID 3812410, 本地 3090

---

| Epoch | mAP | R1 | cipgfr loss | 备注 |
|-------|------|------|------------|------|
| 10 | 38.4% | 51.8% | N/A (warmup) | = PAA |
| 20 | 47.7% | 60.0% | N/A (刚激活) | ≈ PAA |
| 30 | 52.6% | 65.3% | 1.94 | recovery loss 激活后第一个 eval |

| 40 | 55.7% | 68.3% | 2.15 | |
| 50 | 56.8% | 69.4% | 2.10 | 收敛中，cipgfr loss 稳定 |

| 60 | 58.9% | 71.7% | 2.37 | ≈ PAA ep60! |

| 70 | 59.3% | 71.9% | 2.15 | 落后 PAA -1.0% |

### [01:49] Ep79 进行中
- Ep60 ≈ PAA，但 Ep70 开始落后 -1.0%
- cipgfr loss ~2.15，模型在忽略 recovery loss
- ETA ~40min

---
