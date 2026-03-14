# exp052 KP-RPE 训练监控日志

## 实验信息
- **方法**: PSG + Skeleton GCN + KP-RPE (Keypoint Relative Position Encoding)
- **配置**: `configs/occluded_duke/pose_psg_gcn_kprpe.yml`
- **输出**: `log/occluded_duke/exp052_kprpe/`
- **对照**: exp030a (PSG+GCN, equal_concat) 3-seed mean = 60.73% mAP / 72.57% R1
- **核心改动**: 在 Swin Stage 3 attention 中添加 keypoint 相对位置编码
- **启动时间**: 2026-03-14 05:59
- **PID**: 279007

---

### [05:59] 检查点 #1

**状态**: 🟢正常
**进度**: Epoch 1/120 (~0.8%)

| 指标 | 当前值 | 变化趋势 |
|------|--------|----------|
| Total Loss | 20.74 | ↓ 下降中 |
| ID Global | 6.555 | — |
| ID Part | 6.714 | — |
| Tri Global | 12.72 | ↓ |
| Tri Part | 15.50 | ↓ |
| Base LR | 4.76e-05 | Warmup 阶段 |

**观察**: 初始 loss 值与 exp030a baseline 一致，KP-RPE 零初始化确保训练起步正常。
**决策**: 继续，2 分钟后检查
