# exp075 PAA Multi-Seed 验证监控

## 实验信息
- **目标**: 验证 PAA (exp066) 和 PAA+ROA (exp067) 在不同 seed 下的稳定性
- **对照**: exp030a 3-seed mean = 60.73%/72.57%
- **已有单 seed 结果**: PAA seed1234 = 61.6%/74.2%, PAA+ROA seed1234 = 62.0%/73.7%

## 运行实例

### 本地 3090: PAA+ROA seed42
- **启动**: 2026-03-16 09:58, PID 2948756
- **Config**: `pose_psg_gcn_paa_roa.yml`, SOLVER.SEED=42

### 远程 5060 Ti: PAA seed42
- **启动**: 2026-03-16 17:58, PID 10960
- **Config**: `pose_psg_gcn_paa.yml`, SOLVER.SEED=42

### 远程 5060 Ti: PAA seed1234 (已完成)
- **最终结果**: mAP 61.2% / R1 74.3%
- 与本地 3090 (61.6%/74.2%) 一致，Δ<0.4%

---

## 本地 PAA+ROA seed42 训练日志

### Ep10: 37.7%/? (vs seed1234: 38.4%)
seed 差异正常。

### Ep20: 45.9%/?
继续上升。

---

## 远程 PAA seed42 训练日志

### Ep10: 39.3%/? (vs seed1234 remote: 37.9%)
不同 seed 早期略有差异。

---
