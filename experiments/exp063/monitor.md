# exp063 Pose-Token Distillation (PTD) 训练监控

## 实验信息
- **方法**: PSG + PTD (5 part tokens, 2-layer decoder, heatmap distillation)
- **配置**: `configs/occluded_duke/pose_psg_ptd.yml`
- **输出**: `log/occluded_duke/exp063_ptd/`
- **对照**: exp030a (PSG+GCN) 3-seed = 60.73%/72.57%
- **启动时间**: 2026-03-15 07:26
- **PID**: 1261126

---

### [07:37] 检查点 #1 — Epoch 10

| 指标 | exp063 PTD | exp030a GCN |
|------|------------|-------------|
| mAP | 35.5% | 38.2% |
| R1 | 46.7% | 51.3% |

**观察**: PTD 落后 -2.7%/-4.6%。recon loss = 0.000 → heatmap distillation 信号太弱（MSE 在归一化注意力图上量级极小）。Part tokens 缺乏足够的空间定位监督。
**风险**: 如果 heatmap loss 不起作用，PTD 可能退化为 random tokens → 纯靠 ID loss 学习 → 效果差。

---

### [07:56] 检查点 #2 — 修复后重启 (epoch 1)

**修复内容**:
1. DecoderLayer 添加 per-layer Q/K/V projections
2. Heatmap loss 从 MSE 改为 KL divergence，weight 10.0

**修复前**: recon=0.000（MSE 太小，无监督信号）
**修复后**: recon=0.300（KL divergence，有效监督 ✅）

训练正常，等待 epoch 10 评估。

---

### [08:06] 检查点 #3 — Epoch 10（修复后）

| 指标 | exp063 PTD (fixed) | exp063 PTD (pre-fix) | exp030a GCN |
|------|-------------------|---------------------|-------------|
| mAP | 32.0% | 35.5% | 38.2% |
| R1 | 42.8% | 46.7% | 51.3% |
| recon | 0.106 | 0.000 | N/A |

**观察**: 修复后 PTD 反而更低了（-3.5% vs pre-fix）！这是因为 KL divergence loss (recon=0.106) 消耗了部分学习容量来对齐 attention maps 到 heatmap 分布，短期内降低了 ID 分类性能。
- recon loss 从 0.300(ep1) 降到 0.106(ep9)，说明 attention 确实在学习定位
- 但整体性能落后 GCN -6.2% mAP — 差距很大
- Part tokens 需要更长时间来同时学习定位+判别

**决策**: 继续。这个差距在后期是否会缩小是关键。如果 ep30+ 追赶到 -2% 以内，说明 PTD 有潜力。

---

### [08:50] 检查点 #4 — Epoch 20-50 趋势

| Epoch | PTD mAP | GCN mAP | Δ mAP | PTD R1 | GCN R1 | Δ R1 |
|-------|---------|---------|-------|--------|--------|------|
| 10 | 32.0% | 38.2% | -6.2% | 42.8% | 51.3% | -8.5% |
| 20 | 45.2% | 46.8% | -1.6% | 56.2% | 60.9% | -4.7% |
| 30 | 47.5% | 52.2% | -4.7% | 57.1% | 66.0% | -8.9% |
| 40 | 52.2% | 55.6% | -3.4% | 61.0% | 68.6% | -7.6% |
| 50 | 53.8% | 55.7% | -1.9% | 64.0% | 68.8% | -4.8% |

**观察**: mAP 在持续追赶（-6.2→-1.9%），R1 波动但总体缩小（-8.5→-4.8%）。recon loss 从 0.3→0.06，attention maps 在逐步学习定位。
**问题**: R1 持续大幅落后，说明 part tokens 的定位精度不如 GCN 的 bilinear sampling。
**预期最终结果**: mAP 可能追平或微落后，R1 落后 2-4%。整体负面。

---

### [10:05] 最终结果

| Epoch | PTD mAP | GCN mAP | Δ | PTD R1 | GCN R1 | Δ |
|-------|---------|---------|---|--------|--------|---|
| 60 | 55.3% | 57.7% | -2.4% | 64.7% | 70.8% | -6.1% |
| 80 | 56.2% | 59.4% | -3.2% | 65.4% | 72.6% | -7.2% |
| 100 | 56.4% | 60.1% | -3.7% | 65.1% | 73.4% | -8.3% |
| **120** | **56.7%** | **60.5%** | **-3.8%** | **65.3%** | **73.7%** | **-8.4%** |

vs 3-seed mean: **-4.03% mAP / -7.27% R1**

**结论**: PTD 明确失败。Learned part tokens 无法替代 GCN。
- Part tokens 缺乏 GCN 的关键优势：**精确的空间定位**（bilinear sampling at keypoint coords）和 **拓扑先验**（skeleton edges）
- Heatmap KL distillation（recon 0.3→0.07）确实让 tokens 学到了一些定位，但精度远不如直接用关键点坐标
- 这进一步确认了 exp053 XCAD 的教训：**GCN 的 skeleton topology 和 bilinear sampling 是不可替代的**
