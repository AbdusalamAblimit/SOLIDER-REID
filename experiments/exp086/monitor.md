# exp086 PA-PAT (Parallel Augmentation Training) 监控

## 实验信息
- **方法**: PSG + GCN + PAA + 三路并行增强 (full + ROA + forced_RE)
- **配置**: `configs/occluded_duke/pose_psg_gcn_paa_parallel.yml`
- **对照**: exp066 PAA = 61.6%/74.2%, exp079 ROA = 62.0%/73.6%
- **核心改动**: 每个样本 3 个增强版本同时训练，loss 取平均
- **GPU 使用**: ~19GB/24GB (3 forward passes)
- **启动**: 2026-03-17 02:36, PID 3959090, 本地 3090

---

### [02:37] Ep1 训练开始
- Loss 21.2 (3路平均，比单路高正常)
- GPU 19GB，不会 OOM
- 训练速度: ~55s/epoch × 3 forward ≈ 实际每 epoch 更慢

---
