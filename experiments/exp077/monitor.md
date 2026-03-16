# exp077 ST-PAA 训练监控

## 实验信息
- **方法**: PSG + GCN + ST-PAA (Scene+Target concat, 34ch PAA)
- **配置**: `configs/occluded_duke/pose_psg_gcn_paa_st.yml`
- **对照**: exp066 PAA seed1234 = 61.6%/74.2%
- **核心改动**: PAA 输入从 17ch (scene) 改为 34ch (scene+target concat)
- **启动**: 2026-03-16 21:20 (远程 5060 Ti), PID 18500

---

| Epoch | ST-PAA mAP | ST-PAA R1 | PAA ref |
|-------|-----------|----------|---------|
| 10 | 36.3% | 47.6% | 38.4%/51.8% |
| 20 | 47.6% | 61.0% | — |
| 30 | 51.5% | 64.2% | — |

| 40 | 55.8% | 67.7% | — |

### [14:26] Ep47 进行中
- Ep40 eval 55.8%/67.7% — 收敛正常
- 关键对照点: Ep60 (PAA=58.8%) / Ep80 (PAA=61.2%) / Ep120 (PAA=61.6%)
- ETA ~1h

---
