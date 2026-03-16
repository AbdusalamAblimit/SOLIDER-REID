# exp071 Pose-Conditioned LoRA (PCL) 训练监控

## 实验信息
- **方法**: PSG + GCN + PCL (rank=16, ~50K params)
- **对照**: exp066 PAA (feature-independent, 51.8K params) = 61.6%/74.2%
- **核心变量**: adapter 从 feature-independent (PAA) → feature-dependent (PCL)

---
