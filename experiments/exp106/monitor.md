# exp106 PISD (Pose-Informed Self-Distillation) 监控

## 配置
- 基线: exp066 (PSG+GCN+PAA) 61.6%/74.2%
- 改动: image-level pose masking + 二次 forward + cosine self-distillation
- 零新参数（纯训练范式改变）

## 启动确认
- Ep1, 7.9GB (warmup 期间无二次 forward), 57s/epoch
- PISD 激活后（Ep11+）速度预计翻倍（~114s/epoch）

## 待更新...
