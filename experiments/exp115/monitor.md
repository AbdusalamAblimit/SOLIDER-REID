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
