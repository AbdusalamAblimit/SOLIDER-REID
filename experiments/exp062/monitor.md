# exp062 Learned Keypoint Uncertainty (LKU) 训练监控

## 实验信息
- **方法**: PSG + GCN + LKU (learned uncertainty weighting)
- **配置**: `configs/occluded_duke/pose_psg_gcn_lku.yml`
- **输出**: `log/occluded_duke/exp062_lku/`
- **对照**: exp030a (PSG+GCN, kp_weight=confidence) 3-seed = 60.73%/72.57%
- **核心改动**: 学习每个关键点的 uncertainty，用 (1-unc) 调制 confidence 权重
- **启动时间**: 2026-03-15 04:50
- **PID**: 1095245

---

### [04:51] 检查点 #1
**状态**: 🟢正常
**观察**: uncertainty 初始值 0.442（接近 sigmoid 中点 0.5），正常。Loss 量级与 baseline 一致。

---

### [05:01] 检查点 #2 — Epoch 10

| 指标 | exp062 LKU | exp030a |
|------|------------|---------|
| mAP | 36.3% | 38.2% |
| R1 | 48.6% | 51.3% |

**观察**: LKU 在 ep10 落后 -1.9%/-2.7%。uncertainty weighting 在早期扰乱了 GCN 的学习。uncertainty 稳定在 ~0.42（没有 collapse）。等待后期看是否恢复。

---

### [07:00] 最终结果

**最终**: mAP **60.7%** / R1 **71.2%** / R5 84.1% / R10 87.4%

vs exp030a 3-seed mean: **-0.03% mAP / -1.37% R1** — **负面（R1 显著下降）**

Uncertainty 从 0.442 降至 0.229，没有 collapse。但 R1 全程 -1.5~2.5% 落后。

**结论**: Learned Keypoint Uncertainty 负面。额外的 uncertainty head 干扰了 GCN 的 keypoint weighting，产出"更平滑"的 fusion 特征，牺牲了 top-1 精度。第 9 个失败的 GCN branch 训练端修改。
