# exp083 PGFI 训练监控

## 实验信息
- **方法**: PSG + GCN + PAA + PGFI (Pose-Guided Feature Inpainting)
- **配置**: `configs/occluded_duke/pose_psg_gcn_paa_pgfi.yml`
- **对照**: exp066 PAA = 61.6%/74.2%
- **核心改动**: 在 feature map 上做 pose-guided inpainting，恢复遮挡区域特征
- **启动**:
  - 本地 3090: 2026-03-16 22:16, PID 3677709
  - 远程 5060 Ti: 2026-03-17 06:17

---

### [22:18] Ep2 训练开始
- 本地 Loss 正常下降
- 注意 tri_part 6.49（比 GCN 的 ~17 低），PGFI 影响了 feature map 分布
- 远程 Ep1 也正常

---
