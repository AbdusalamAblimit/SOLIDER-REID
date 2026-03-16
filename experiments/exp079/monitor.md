# exp079 PSG+GCN+ROA (无PAA) 训练监控

## 实验信息
- **方法**: PSG + GCN + ROA（无 PAA）
- **配置**: `configs/occluded_duke/pose_psg_gcn_roa_nopaa.yml`
- **对照**: exp030a 3-seed = 60.73%/72.57%, exp058 ROA+PAA: 61.8%/72.8%
- **核心问题**: ROA 在不带 PAA 的基础框架上是否也有效？
- **启动**: 2026-03-17 00:22 (远程 5060 Ti), PID 24678

---

### Ep10: 38.3%/51.0% — 与 exp030a baseline 一致

| Epoch | mAP | R1 | R10 |
|-------|------|------|------|
| 10 | 38.3% | 51.0% | 73.2% |
| 20 | 48.3% | 61.5% | 80.8% |
| 30 | 53.5% | 66.0% | 83.7% |

### [17:05] Ep30 eval
- 53.5%/66.0% — 与 exp030a baseline 相当
- ETA ~1h

---
