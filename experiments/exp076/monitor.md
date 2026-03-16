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
- id_global/id_part ~6.55 (随机初始化后的正常值)
- 确认 TDPC 模块已加载，无报错

---
