# exp037: Learnable Keypoint Attention (LKA) — 监控日志

## 实验概述
- **目的**: 用可学习 MLP 替换固定置信度加权，发现最优关键点重要性分配
- **Base**: exp035a (PSG + GCN, score weight, equal_concat) = 61.1% mAP / 73.8% R1
- **变量**: 仅增加 POSE_KP_LEARNABLE_ATTN=True (~600 params)
- **PID**: 3383953
