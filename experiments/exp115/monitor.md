# exp115 监控

## 实验信息
- 方法: Freeze-Later Reliable SCKD
- 类型: 训练端单变量改进
- 运行位置: 远程 5060 Ti
- 主配置: `exp112_sckd_up07`
- 核心变量: `POSE_SCKD_UPDATE_STOP_EPOCH = 30`

## 启动记录

### [2026-03-19 19:05] 实验准备
- 启动原因:
  1. 远程当前空闲，可用于并行验证 `teacher stability`
  2. `exp114` 测的是最强冻结版本 `freeze20`
  3. 还需要一个互补对照，区分 “冻结本身有害” 和 “冻结时机过早”
- 当前执行内容:
  1. 保持 `update_thr=0.7`
  2. 仅把 `POSE_SCKD_UPDATE_STOP_EPOCH` 设为 `30`
  3. 在远程 5060 Ti 后台启动
- 当前判断: 待启动
- 原因:
  - 这是与 `exp114` 最互补、最省时间的并行单变量

### [2026-03-20 03:01] 启动确认（远程 5060 Ti）
- 运行位置: 恒源云 5060 Ti
- 远程仓库: 已同步到 `94fd7d1`
- 启动方式: 后台 `nohup`
- 输出目录: `log/occluded_duke/exp115_sckd_up07_freeze30`
- nohup 日志: `log/occluded_duke/exp115_sckd_up07_freeze30/remote_nohup.log`
- 关键确认:
  1. 配置已生效：`update_thr=0.7, stop_epoch=30`
  2. 日志已打印：
     - `[SCKD] enabled: weight=0.5, warmup=20, low_thr=0.3, update_thr=0.7, mom=0.9, stop_epoch=30`
  3. GPU 已占用约 `6.7GB`
  4. `Epoch[1] Iter[60/227] Loss: 19.126`
- 当前判断: 继续
- 原因:
  - 现在形成了本地 `freeze20` + 远程 `freeze30` 的并行对照
