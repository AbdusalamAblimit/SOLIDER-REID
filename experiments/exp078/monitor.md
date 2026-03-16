# exp078 APG (Adaptive PAA Gate) 训练监控

## 实验信息
- **方法**: PSG + GCN + PAA + Adaptive Gate
- **配置**: `configs/occluded_duke/pose_psg_gcn_paa_apg.yml`
- **对照**: exp066 PAA seed1234 = 61.6%/74.2%
- **核心改动**: PAA 输出乘 adaptive gate (sigmoid(MLP(hm_pool)))，学习抑制单人图的 PAA
- **数据驱动假设**: subset analysis 显示 PAA 在单人图 R1 退化 -1.61%
- **启动**: 2026-03-16 14:38, PID 3240372, 本地 3090

---

### [14:39] Ep1 训练开始
- Loss 正常下降 (18.0 → warmup 期正常)
- 与 exp066/exp076 的 Ep1 一致

---
