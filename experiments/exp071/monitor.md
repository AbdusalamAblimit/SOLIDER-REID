# exp071 Pose-Conditioned LoRA (PCL) 训练监控

## 实验信息
- **方法**: PSG + GCN + PCL (rank=16, ~50K params)
- **对照**: exp066 PAA (feature-independent, 51.8K params) = 61.6%/74.2%
- **核心变量**: adapter 从 feature-independent (PAA) → feature-dependent (PCL)

- **启动**: 2026-03-16 01:57, PID 2447761

---

### Ep10: 39.2%/52.1% (vs PAA: 38.4%/51.8%)
PCL 略超 PAA +0.8%/+0.3%。早期微正信号。

### Ep20: 46.7%/59.9%

### Ep30: 52.8%/64.6%

### Ep40: 56.4%/68.5%

### Ep50: 56.9%/69.1%

### Ep60: 58.3%/70.9%

### Ep70: 58.6%/70.3%

### Ep80: 59.7%/71.2%

### Ep90: 60.0%/71.7%

### Ep100: 60.3%/71.9%

### Ep110: 60.7%/71.9%

### Ep120（最终）: 60.7%/72.0%

---

## 最终结论

| 指标 | PAA (exp066) | PCL (exp071) | 差异 |
|------|-------------|-------------|------|
| mAP | 61.6% | 60.7% | -0.9% |
| R1 | 74.2% | 72.0% | -2.2% |
| R5 | 85.4% | 84.6% | -0.8% |
| R10 | 88.4% | 88.1% | -0.3% |

**结论**: PCL (feature-dependent LoRA) 全面劣于 PAA (feature-independent adapter)。说明 pose injection 的最优形式是 feature-independent addition，而不是 feature-dependent modulation。PAA 的简洁性（不依赖当前特征内容）反而是优势——pose 信号不需要"看"当前特征就能有效注入。

**论文价值**: 支撑 PAA 设计选择的消融证据——"feature-independent adapter 优于 feature-dependent LoRA"。
