# exp188 OA-SD (Occlusion-Asymmetric Self-Distillation) 监控

配置: exp176 + POSE_OA_SD=True (EMA teacher, decay=0.999)
- Teacher: EMA of student, 看 clean image (pre-PLBOA)
- Student: 看 occluded image (post-PLBOA)
- Distillation: per-token cosine distance

## 检查点

### [20:36] 检查点 #1

**状态**: 正常
**进度**: Epoch 1/120

| 指标 | 当前值 | 趋势 |
|------|--------|------|
| Total Loss | 11.9 | 下降中 |
| SupCon | 4.51 | 下降 |
| ID Global | 6.554 | 稳定 |
| Tri Global | 11.1 | 快速下降 |
| **oa_sd** | **0.475** | 上升（正常：EMA 滞后导致 teacher-student 差距增大）|
| LR | 4.76e-05 | warmup |

**观察**: OA-SD 正常工作。oa_sd loss 从 0.40 上升到 0.48（EMA teacher 尚未充分更新，student 快速学习导致差距增大）。
**Bug 修复**: teacher forward 时需 train() mode（否则返回 test-path 的 2 值）。已修复。
**决策**: 继续

### [20:43] 检查点 #2

**状态**: 正常
**进度**: Epoch 6/120

| 指标 | 当前值 | 趋势 |
|------|--------|------|
| SupCon | 3.83 | ↓ 正常下降 |
| oa_sd | **0.205** | ↓ 快速下降（从 0.475→0.205）|
| Tri Global | 1.03 | ↓ |
| LR | 2.46e-04 | warmup |

**观察**: oa_sd loss 快速下降（0.475→0.205 = -57%），说明 EMA teacher 和 student 正在快速收敛。好信号。
Remote exp189 完成: 63.3/73.7 (vis-weighted SupCon = 负向 -0.8/-1.8 vs uniform)。
**决策**: 继续
