# exp035: Visibility 最小闭环消融 — 监控日志

## 实验概述
- **目的**: 对比 4 种 keypoint pooling 权重模式
- **Base**: exp030a (PSG + GCN, equal_concat)
- **对照**: exp030a 3-seed mean mAP 60.73%, R1 72.57%

## 子实验列表
| ID | 权重模式 | Config | 状态 |
|----|---------|--------|------|
| 035a | score (baseline) | exp035a_kpw_score.yml | 待训练 |
| 035b | score * visibility | exp035b_kpw_score_visibility.yml | 待训练 |
| 035c | visibility only | exp035c_kpw_visibility.yml | 待训练 |
| 035d | binary visibility | exp035d_kpw_binary_visibility.yml | 待训练 |

---
