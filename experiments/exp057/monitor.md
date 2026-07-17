# exp057 Keypoint Dissimilar Loss (KDL) 训练监控

## 实验信息
- **方法**: PSG + GCN + KDL (weight=0.1)
- **配置**: `configs/occluded_duke/pose_psg_gcn_kdl.yml`
- **输出**: `log/occluded_duke/exp057_kdl/`
- **对照**: exp030a (PSG+GCN) 3-seed mean = 60.73% mAP / 72.57% R1
- **核心改动**: 加 Keypoint Dissimilar Loss 防止 GCN 17 个关键点特征坍缩
- **启动时间**: 2026-03-14 17:55 (重启后，修复了 kp_data 传递 bug)
- **PID**: 703348

---

### [17:56] 检查点 #1
**状态**: 🟢正常
**进度**: Epoch 1/120

**观察**: KDL loss 初始值 0.556（余弦相似度在 0~1 间），正常。Loss 总量级与 exp030a 一致。训练速度正常。

---

### [18:06] 检查点 #2 — Epoch 10

| 指标 | exp057 KDL | exp030a |
|------|------------|---------|
| mAP | 38.4% | 38.2% |
| R1 | 51.3% | 51.3% |

KDL loss 趋势: 0.556(ep1) → 0.777(ep5)，在增加而非减少。说明 GCN 特征在训练初期自然趋同，KDL 在对抗这个趋势但 weight=0.1 可能不够强。等待后期结果。

---

### [20:00] 检查点 #3 — 最终结果

**最终**: mAP **61.0%** / R1 **73.3%** / R5 84.6% / R10 87.9%

vs exp030a 3-seed mean: **+0.27% mAP / +0.73% R1** — 方差范围内

**KDL loss 趋势**: 0.556(ep1) → 0.777(ep5) → 0.740(ep20) → 稳定。weight=0.1 使 KDL 对总 loss 贡献仅 ~0.07，可能过弱。

**结论**: KDL 中性。关键点特征的"坍缩"（高余弦相似度）可能不是问题——GCN 特征高度相似可能是因为它们都编码同一个人的 ID 信息，这本身是有益的。推散特征反而可能降低 ID 判别力。

**第 6 个中性/失败的 auxiliary loss 实验**。训练端 loss 修改方向已完全耗尽。
