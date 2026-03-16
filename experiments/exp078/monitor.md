# exp078 APG (Adaptive PAA Gate) 训练监控

## 实验信息
- **方法**: PSG + GCN + PAA + Adaptive Gate
- **配置**: `configs/occluded_duke/pose_psg_gcn_paa_apg.yml`
- **对照**: exp066 PAA seed1234 = 61.6%/74.2%
- **核心改动**: PAA 输出乘 adaptive gate (sigmoid(MLP(hm_pool)))，学习抑制单人图的 PAA
- **数据驱动假设**: subset analysis 显示 PAA 在单人图 R1 退化 -1.61%
- **启动**: 2026-03-16 14:38, PID 3240372, 本地 3090

---

### [14:49] Ep10 评估
- **mAP 37.5% / R1 49.5%** vs PAA ep10: 38.4%/51.8%
- **Δ -0.9%/-2.3%** — 早期落后，可能因为 gate=0.5 减半了 PAA
- 这不一定是坏信号——gate 需要学习才能区分单人/多人
- 关键对照: Ep60+

| Epoch | APG mAP | APG R1 | PAA ref mAP | PAA ref R1 |
|-------|---------|--------|-------------|-----------|
| 10 | 37.5% | 49.5% | 38.4% | 51.8% |
| 20 | 47.0% | 59.8% | — | — |

| 30 | 53.2% | 66.4% | — | — |

| 40 | 55.6% | 68.8% | — | — |

### [15:20] Ep40 — 与 TDPC 持平
- APG Ep30=53.2% vs TDPC=52.1% (+1.1%); Ep40=55.6% vs TDPC=56.3% (-0.7%)
- APG 和 TDPC 在前期几乎一样（两者都以 PAA 为基础）
- 关键对照: Ep60 (PAA=58.8%) / Ep80 (PAA=61.2%)

---
