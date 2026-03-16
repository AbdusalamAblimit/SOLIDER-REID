# exp076 TDPC 训练监控

## 实验信息
- **方法**: PSG + GCN + PAA + TDPC (Target-Distractor Differential Adapter)
- **配置**: `configs/occluded_duke/pose_psg_gcn_paa_tdpc.yml`
- **对照**: exp066 PAA seed1234 = 61.6%/74.2%, exp030a 3-seed = 60.73%/72.57%
- **核心改动**: 在 PAA 之后加 TDDA 模块，输入 H_target - H_distractor (17ch → 32 → 768, zero-init, ~51.8K params)
- **启动**: 2026-03-16 12:05, PID 3095209

---

### [12:06] Ep1 训练开始
- Loss 正常下降中 (18.1 → warmup 期正常)
- 确认 TDPC 模块已加载，无报错

### [12:16] Ep10 首次评估
- **mAP 38.3% / R1 52.0%**
- vs exp066 PAA ep10: 38.4%/51.8% → **Δ -0.1%/+0.2%，完全一致**
- zero-init 正确工作，TDDA 模块尚未开始贡献

### [12:27] Ep20 评估
- **mAP 47.5% / R1 60.6%**
- vs exp066 PAA seed1234 ep20: 无精确数据（ep10=38.4%, 无 ep20 记录）
- 收敛正常，LR 到 0.74e-3 (warmup 结束)

| Epoch | TDPC mAP | TDPC R1 | PAA ref mAP | PAA ref R1 |
|-------|----------|---------|-------------|-----------|
| 10 | 38.3% | 52.0% | 38.4% | 51.8% |
| 20 | 47.5% | 60.6% | — | — |
| 30 | 52.1% | 64.5% | — | — |

| 30 | 52.1% | 64.5% | — | — |
| 40 | 56.3% | 68.6% | — | — |

| 50 | 57.8% | 69.4% | — | — |

### [12:58] Ep51 进行中
- Ep50 eval 57.8%/69.4%，收敛正常
- 从 Ep60 开始与 PAA exp066 seed1234 精确对比
  - exp066 ep60 = 58.8%/72.0%
  - exp066 ep80 = 61.2%/74.4%
  - exp066 ep120 = 61.6%/74.2%
- ETA ~1h5m

---
