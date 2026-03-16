# exp081 PQTD 训练监控

## 实验信息
- **方法**: PSG + PAA + PQTD (Pose-Query Transformer Decoder, 替代 GCN)
- **配置**: `configs/occluded_duke/pose_psg_paa_pqtd.yml`
- **对照**: exp066 PAA (PSG+GCN+PAA) = 61.6%/74.2%
- **核心改动**: 用 3-layer transformer decoder + 5 pose-guided queries 替代 2-layer GCN
- **参数**: ~2.5M (vs GCN ~400K)
- **启动**: 2026-03-16 17:01, PID 3378400, 本地 3090

---

### [17:03] Ep2 进行中
- Loss 9.5 (正常下降), id_part 6.55 (decoder 初始化阶段，还没开始学)
- GPU 78%, 8054MiB
- 审查在后台进行中
- 注意: PQTD 的 id_part 收敛可能比 GCN 慢（decoder 更复杂）

---
