# exp382 Video TAPF 监控记录

## 2026-07-17 14:38：专项查新与数据审计

- 状态：`NO-GO / 未下载 / 未实现 / 未训练`
- 4090：`2 MiB / 0%`，无训练进程。
- 数据：`/home/afr/datasets/AG-VPReID.VIR` 为 `4.0K` 空目录；未发现 MARS、DukeMTMC-VideoReID、iLIDS-VID、PRID2011。
- 查新：
  - GAE-Net 已覆盖训练期 gait+RGB 视频教师、RGB-only 视频学生及局部互补蒸馏；
  - PAFormer 已覆盖 pose-supervised、pose-free inference；
  - KPRTrack 已覆盖 tracklet 同部位 moving average；
  - STMN/TF-CLIP/PSTA/CTL/GRL 与 AG-VPReID 系列已覆盖遮挡、干扰、memory 和多粒度 temporal modeling。
- 判断：候选 temporal pose state 仍无法从“结构特权蒸馏 + 同部位 tracklet 聚合 + temporal memory”近邻中形成足够清楚的独立方法差分。
- 动作：不下载大数据，不启动 4090，不写训练实现；Video TAPF 只保留为未来应用扩展候选。

## 结论边界

本轮 NO-GO 只否定 Video TAPF 作为新的独立 headline，不否定：

1. 单图完整 `anchor+PSG` 方法已经取得的三骨干正向证据；
2. 将 D0 迁移到视频作为论文外部验证的潜在价值；
3. 未来若出现真正不同的跨帧可测变量，再以新查新门禁重新开启。

本轮没有 checkpoint、日志或训练指标，不得把任何旧单图数字伪装成视频结果。
