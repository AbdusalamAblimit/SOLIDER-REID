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

### [00:57] Ep30 — recovery loss 激活后首次 eval
- cipgfr loss = 1.94 且在上升（1.84→1.94），这可能是正常的（target 变好→gap 暂时增大）
- 也可能是负信号（features 过拟合到各自 view）
- 需要 Ep60+ 判断趋势

---
