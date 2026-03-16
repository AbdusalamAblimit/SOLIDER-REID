# exp077 ST-PAA 训练监控

## 实验信息
- **方法**: PSG + GCN + ST-PAA (Scene+Target concat, 34ch PAA)
- **配置**: `configs/occluded_duke/pose_psg_gcn_paa_st.yml`
- **对照**: exp066 PAA seed1234 = 61.6%/74.2%
- **核心改动**: PAA 输入从 17ch (scene) 改为 34ch (scene+target concat)
- **启动**: 2026-03-16 21:20 (远程 5060 Ti), PID 18500

---

### [21:35] Ep10 首次评估 ⚠️
- **mAP 36.3% / R1 47.6%**
- vs exp066 PAA ep10: 38.4%/51.8% → **Δ -2.1%/-4.2%**
- 早期差距较大，可能原因：34ch Conv2d 随机初始化改变了优化路径
- 但 zero-init 输出层保证初始输出为 0，差距应随训练缩小
- 需继续观察到 Ep60+

---
