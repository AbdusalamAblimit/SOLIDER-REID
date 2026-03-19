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

### [2026-03-19 19:13] 检查点 #1 — Epoch 10-32
- 结果:
  - `ep10 = 38.3% / 51.3% / 68.8% / 74.6%`
  - `ep20 = 47.1% / 59.7% / 75.4% / 80.2%`
  - `ep30 = 52.6% / 65.2% / 78.9% / 84.1%`
- 对照:
  - `exp110` 同期约为：
    - `38.3 / 51.3`
    - `47.1 / 59.7`
    - `52.6 / 65.4`
  - `exp113` 同期约为：
    - `38.3 / 51.3`
    - `47.1 / 59.7`
    - `52.6 / 65.3`
- 当前观察:
  1. 到 `ep30` 为止，冻结 teacher 没有带来验证上的立刻收益，曲线与 `exp110/113` 基本重合，`R1` 还略低 `0.1~0.2`
  2. 但机制层面的变化已经很明显：
     - `exp113` 在线 teacher 在 `ep30` 时：
       - `sckd = 0.1925`
       - `count = 436.6`
       - `cos = 0.8084`
     - `exp114 freeze20` 在 `ep30` 时：
       - `sckd = 0.1781`
       - `count = 298.8`
       - `cos = 0.8225`
  3. 这说明冻结 teacher 的确抑制了 hardening：
     - raw `sckd` 更低
     - `sckd_cos` 更高
     - `count` 不再像在线版本那样持续上升
  4. 日志里 `count` 仍有批间波动是正常现象，因为这里记录的是“当前 batch 命中的 prototype 平均 count”，不是全局 bank 总量
- 当前判断: 继续
- 原因:
  - 当前已经确认机制被改动成功；下一步关键是看 `ep40/50` 时这种更稳定的 teacher 是否会转成验证优势
