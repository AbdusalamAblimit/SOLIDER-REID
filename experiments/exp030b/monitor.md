# exp030b 训练监控日志

## 实验配置
- **Config**: `configs/occluded_duke/pose_psg_gcn_noscale.yml`
- **Output**: `./log/occluded_duke/exp030b_psg_gcn_noscale`
- **核心变量**: POSE_PART_WEIGHT=0.01 (w_g≈1.0, w_p≈0.01)
- **对照**: exp007 (PSG 1.0x loss, mAP 58.3%), exp030a (PSG+GCN 0.5x, mAP 61.1% eq)
- **预期**: global ≈ 58.3% (PSG 基线), equal_concat > global (GCN 特征贡献)

---
