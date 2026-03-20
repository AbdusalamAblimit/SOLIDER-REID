# exp121 监控

## 实验信息
- 方法: SCRD Freeze-30
- 类型: 训练端单变量改进
- 运行位置: 远程 5060 Ti
- 主配置: `exp120_scrd`
- 核心变量: `POSE_CSRD_ST_UPDATE_STOP_EPOCH = 30`
- 输出目录: `log/occluded_duke/exp121_scrd_freeze30`

## 启动记录

### [2026-03-20 14:56] 实验准备

- 启动原因:
  1. 本地 `exp120` 已正确重启并通过 warmup 早期稳定性检查
  2. 当前最有信息量的远程并行对照，不是重复一份 `exp120`，而是测试 support-complete teacher 的稳定化
  3. `freeze30` 是当前最自然的单变量：既不回退到旧 `SCKD`，也不打断 `exp120` 的问题链
- 当前执行内容:
  1. 将当前分支推送到远程仓库
  2. 远程拉取最新 `exp/pose_heatmap`
  3. 在恒源云 5060 Ti 后台启动 `exp121`
- 当前判断: 待启动
- 原因:
  - 需要形成“本地 online support-complete teacher + 远程 freeze30 support-complete teacher”的并行对照

### [2026-03-20 15:27] 首次启动失败并立即修正

- 异常:
  1. 第一次远程启动时，把 `nohup` 输出重定向到了尚未创建的目录
  2. 实际报错：
     - `log/occluded_duke/exp121_scrd_freeze30/remote_nohup.log: No such file or directory`
- 处理:
  1. 不保留该次结果
  2. 在启动脚本中补 `mkdir -p log/occluded_duke/exp121_scrd_freeze30`
  3. 重新后台启动
- 当前判断: 继续
- 原因:
  - 这是启动脚本问题，不是实验机制问题；修正后可继续按单变量方案执行

### [2026-03-20 15:30] 启动确认（远程 5060 Ti）

- 运行位置: 恒源云 5060 Ti
- 远程仓库: 已同步到 `c4ea76b`
- 启动方式: 后台 `nohup`
- 输出目录: `log/occluded_duke/exp121_scrd_freeze30`
- nohup 日志: `log/occluded_duke/exp121_scrd_freeze30/remote_nohup.log`
- 关键确认:
  1. 配置已生效：`POSE_CSRD_SUPPORT_TEACHER=True`
  2. freeze 变量已生效：`stop_epoch=30`
  3. 日志已打印：
     - `[CSRD-ST] enabled: low_thr=0.3, update_thr=0.7, mom=0.9, min_count=1, stop_epoch=30`
  4. GPU 已占用约 `6.7GB`，利用率约 `86%`
- 当前判断: 继续
- 原因:
  - 现在已经形成了本地 `exp120 online teacher` + 远程 `exp121 freeze30 teacher` 的并行对照

### [2026-03-20 15:32] 检查点 #1 — Epoch 10

- 结果:
  - `ep10 = 38.9% / 53.4% / 68.1% / 74.2%`
- 对照:
  - 本地 `exp120 ep10 = 39.8 / 52.9`
  - `exp119 ep10 = 39.8 / 52.9`
- 当前观察:
  1. 远程 `freeze30` 在 warmup 阶段没有异常，已正常完成首次验证
  2. 当前还看不到 freeze30 的真实影响，因为 `CSRD` / teacher bank 都尚未激活
  3. 相比本地，远程表现为 `mAP -0.9 / R1 +0.5`，更像跨硬件 / early-phase 差异，不宜过度解释
- 当前判断: 继续
- 原因:
  - `exp121` 的真正信息量要从 `epoch 21+` 才开始出现，`ep10` 只说明远程启动健康
