# exp255 Small GCN hidden 512 + 2-stage PSG + LGPA-D + OA-SD 监控

配置: Swin-Small + 2-stage PSG + LGPA-D + GCN hidden=512 + OA-SD + PLBOA + WITH_CP
环境: 远程 5060Ti
对照: exp249 (Small GCN 256, 1-stage): 71.9/81.8, MaxSim 73.3/83.2
目标: 推 Small mAP 向 75%

## 检查点

### [01:20] 检查点 #1 — 启动成功

ep1 iter 40. 训练正常启动。
