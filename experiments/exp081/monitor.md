# exp081 PQTD 训练监控

## 实验信息
- **方法**: PSG + PAA + PQTD (Pose-Query Transformer Decoder, 替代 GCN)
- **配置**: `configs/occluded_duke/pose_psg_paa_pqtd.yml`
- **对照**: exp066 PAA (PSG+GCN+PAA) = 61.6%/74.2%
- **核心改动**: 用 3-layer transformer decoder + 5 pose-guided queries 替代 2-layer GCN
- **参数**: ~2.5M (vs GCN ~400K)
- **启动**: 2026-03-16 17:01, PID 3378400, 本地 3090

---

### [17:01] Ep1 训练开始
- Loss 正常下降
- GPU 78%, 8054MiB (比 GCN 多~500MB，可接受)
- 审查在后台进行中

---
