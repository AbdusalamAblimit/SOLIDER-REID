# exp077 ST-PAA 训练监控

## 实验信息
- **方法**: PSG + GCN + ST-PAA (Scene+Target concat, 34ch PAA)
- **配置**: `configs/occluded_duke/pose_psg_gcn_paa_st.yml`
- **对照**: exp066 PAA seed1234 = 61.6%/74.2%
- **核心改动**: PAA 输入从 17ch (scene) 改为 34ch (scene+target concat)
- **启动**: 2026-03-16 21:20 (远程 5060 Ti), PID 18500

---

### [21:21] Ep1 训练开始
- Loss 正常下降 (17.8 → warmup 期正常)
- 与 exp066/exp076 的 Ep1 一致

---
