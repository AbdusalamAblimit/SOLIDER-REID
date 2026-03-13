# exp050: PAMC (Pose-Aware Masking Consistency) 训练监控

## 实验配置
- 方法: PSG + GCN + PAMC（在 exp030a 基础上增加 pose-aware masking consistency loss）
- Config: `configs/occluded_duke/pose_psg_gcn_pamc.yml`
- 对照: exp030a (PSG+GCN, equal_concat) = 60.73% mAP (3-seed mean)
- PAMC 参数: weight=0.5, warmup=10, proj_dim=2048
- 额外参数: ~3.15M (projector MLP)

---
