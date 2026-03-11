# exp021: Content-Adaptive PSG (CAPSG) — 训练监控

## 实验概要
- **方法**: CAPSG — PSG gate 同时依赖 pose heatmap 和当前特征内容
- **Config**: configs/occluded_duke/pose_capsg.yml
- **Output**: log/occluded_duke/exp021_capsg/
- **PID**: 2350884
- **开始时间**: 2026-03-11 00:24
- **对照**: exp007 PSG (mAP 58.3%, R1 67.9%)
- **核心假设**: content-dependent gate 比 pose-only gate 更精细

---
