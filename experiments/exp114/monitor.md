# exp114 监控

## 实验信息
- 方法: Freeze-After-Warmup Reliable SCKD
- 类型: 训练端单变量改进
- 主配置: `exp112_sckd_up07`
- 核心变量: `POSE_SCKD_UPDATE_STOP_EPOCH = 20`

## 启动记录

### [2026-03-19 19:00] 实验准备
- 启动原因:
  1. `exp112` 说明提升 teacher purity 只有弱正向
  2. `exp113` 诊断表明当前更可疑的问题是 bank 持续增长导致 teacher 逐步变硬
  3. 需要把 “non-stationary teacher” 从解释升级成可验证的单变量
- 当前执行内容:
  1. 保持 `exp112` 的 `update_thr=0.7`
  2. 新增 `POSE_SCKD_UPDATE_STOP_EPOCH = 20`
  3. 让 bank 在 `warmup` 后停止更新，只保留固定 teacher
- 当前判断: 待启动
- 原因:
  - 这是当前最直接、最贴近核心机制的下一步

### [2026-03-19 18:40] 启动确认
- 运行位置: 本地 3090
- 输出目录: `log/occluded_duke/exp114_sckd_up07_freeze20`
- 关键确认:
  1. 配置已生效：`update_thr=0.7, stop_epoch=20`
  2. 日志已打印：
     - `[SCKD] enabled: weight=0.5, warmup=20, low_thr=0.3, update_thr=0.7, mom=0.9, stop_epoch=20`
  3. GPU 已空闲后重新占用，本轮为新的独立训练
- 当前判断: 继续
- 原因:
  - 当前最关键的是看 `ep10/20/30` 是否与 `exp112` 接近，以及 `epoch 21+` 后冻结 teacher 是否改变验证走势
