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

### [01:19] Ep50
- cipgfr loss 从 1.84→2.10 后稳定在 ~2.1
- 整体表现与 PAA 变体（TDPC/APG）同水平
- 关键对照: Ep60 (PAA=58.8%)

---
