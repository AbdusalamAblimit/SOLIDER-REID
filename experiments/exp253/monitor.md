# exp253 Tiny 3-Stage PSG (Stage 1+2+3, 无 PAA) + LGPA-D+GCN 监控

配置: Swin-Tiny + PSG Stage1+2+3 (无 PAA) + LGPA-D (detach) + GCN (detach) + OA-SD + PLBOA(0.7) + WITH_CP
环境: 远程 5060Ti
对照: exp246b (Stage3-only PSG): 65.5/77.2 | exp251 (Stage2+3 PSG+PAA): 65.2/76.2

## 检查点

### [03:38] 检查点 #1 — 启动成功

ep1 iter 80: Loss 16.17, GPU 5966/16311 MiB. 训练正常。
