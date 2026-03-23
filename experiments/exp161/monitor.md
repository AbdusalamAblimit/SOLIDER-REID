# exp161 STD-PR 监控

## 实验信息
- 方法: Structural Token Decomposition with Pose-guided Routing
- 类型: 范式级创新（spatial tokens → structural body-part tokens）
- 基线: exp030a-eq (60.73% mAP 3-seed mean)
- 运行: 本地 3090（等 seed42 完成后启动）
- CHECKPOINT_PERIOD: 20

## 核心关注指标
- equal_concat mAP/R1（与 GCN branch 直接对比）
- token_norm（structural tokens 的范数，监控是否有 collapse）

## 与历史 decoder 实验的关键区别
- exp063 PTD: 2 层 decoder, dim=256, 120ep 不够 → mAP 56.7%
- exp081 PQTD: 3 层 decoder, dim=256, 120ep 不够 → mAP 56.9%
- **exp161 STD-PR**: 2 层 cross-attn, **dim=768**(不降维), **pose heatmap additive bias**
- 如果 dim=768 + pose bias 能让 cross-attn 收敛更快 → 有可能在 120ep 内有效
