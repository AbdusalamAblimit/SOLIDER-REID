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
