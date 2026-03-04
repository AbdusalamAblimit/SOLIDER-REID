# 实验结果总表

## 数据集：Occluded-Duke（唯一实验数据集）

### 历史实验（Pose-Swin 双分支系列）

| ID | 方法 | mAP | R-1 | R-5 | R-10 | 备注 |
|----|------|-----|-----|-----|------|------|
| E0 | Baseline (SOLIDER Swin-Tiny, no pose) | 55.2% (G) | 65.5% (G) | — | — | 基准 |
| E1 | Pose-Swin 双分支 (scale=1.0) | 59.0% (C) | 68.6% (C) | — | — | 主要来自参数翻倍 |
| E7 | 双分支 no-pose (scale=0.0) | 58.7% (C) | 68.8% (C) | — | — | 证明 pose gating 仅+0.3% |
| E1c | Pose-Swin (scale=0.3) | 59.0% (C) | 68.6% (C) | — | — | |

注: G=Global, C=Concat(global+local)

### PAMS 系列

| ID | 方法 | Epoch | global mAP/R1 | parts mAP/R1 | 状态 |
|----|------|-------|--------------|-------------|------|
| PAMS-v8 | L2norm + SoftMargin | 30/120 | 45.8%/55.5% | 32.1%/46.4% | 训练稳定，未完成 |
| PAMS-v9 | v8 + per-part ID loss | — | — | — | 待测试 |

### 新实验系列 (OA-PAMS)

| ID | 方法 | mAP | R-1 | R-5 | R-10 | FLOPs | 推理速度 | 备注 |
|----|------|-----|-----|-----|------|-------|----------|------|
| exp001 | PAMS v9 baseline (120ep) | — | — | — | — | — | — | 待跑 |
| exp002 | Swin-Tiny baseline (120ep) | — | — | — | — | — | — | 待跑 |
| exp003 | PAMS + Soft BPA | — | — | — | — | — | — | 待跑 |
| exp004 | PAMS + NFC 后处理 | — | — | — | — | — | — | 待跑 |
| exp005 | OA-PAMS (全部组件) | — | — | — | — | — | — | 待跑 |
