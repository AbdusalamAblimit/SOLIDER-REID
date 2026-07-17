# exp061 GCN Keypoint Dropout (GKD) 训练监控

## 实验信息
- **方法**: PSG + GCN + GKD (30% keypoint dropout, no reconstruction loss)
- **配置**: `configs/occluded_duke/pose_psg_gcn_gkd.yml`
- **输出**: `log/occluded_duke/exp061_gkd/`
- **对照**: exp030a (PSG+GCN, 无 dropout) 3-seed = 60.73%/72.57%
- **核心改动**: 训练时随机 mask 30% GCN 输入特征，无额外 loss
- **启动时间**: 2026-03-15 02:36
- **PID**: 944645

---

### [02:37] 检查点 #1
**状态**: 🟢正常
**观察**: Loss 正常，sgmkc (reconstruction) loss=14.4 但 weight=0 不参与优化。GCN 在 30% masked 输入上学习 ID classification。

---

### [02:46] 检查点 #2 — Epoch 10

| 指标 | exp061 GKD | exp030a |
|------|------------|---------|
| mAP | 38.5% | 38.2% |
| R1 | 52.1% | 51.3% |

**观察**: GKD 微弱领先（+0.3%/+0.8%），dropout 没有造成负面影响。

---

### [04:44] 最终结果

**最终**: mAP **60.8%** / R1 **73.0%** / R5 84.3% / R10 87.8%

vs exp030a 3-seed mean: **+0.07% mAP / +0.43% R1** — 完全中性

**结论**: GCN Keypoint Dropout 无效。30% 关键点 dropout 没有提供额外的遮挡鲁棒性正则化。GCN 已经通过 skeleton graph propagation 天然地处理了部分缺失关键点。
