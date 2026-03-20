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
