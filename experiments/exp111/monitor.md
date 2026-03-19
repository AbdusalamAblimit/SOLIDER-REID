# exp111 监控

## 实验信息
- 方法: Reliable-Support SCKD
- 类型: 训练端单变量改进
- 主基线: `exp110_sckd`
- 核心变量: `POSE_SCKD_MIN_COUNT = 4`

## 启动记录

### [2026-03-19 18:28] 实验启动
- 启动原因:
  1. `exp110` 已在单 seed 上给出弱正向信号：`61.2 / 73.7`
  2. 当前最可疑瓶颈是 teacher 可靠性，而不是蒸馏是否应存在
  3. `MIN_COUNT=1` 过于宽松，不足以体现真正的 multi-view support
- 当前执行内容:
  1. 保持 `exp110` 其余配置不变
  2. 仅将 `POSE_SCKD_MIN_COUNT` 提高到 `4`
  3. 重点关注 `epoch > 20` 后 `sckd` 曲线与 `ep40/60` 验证
- 当前判断: 继续
- 原因:
  - 这是最干净、最贴近当前机制瓶颈的单变量推进

### [2026-03-19 18:29] 检查点 #1 — Epoch 1-3
- 训练状态: 正常
- 关键日志:
  - `Epoch 1 done`, `59.81s/epoch`, `ETA 1h58m`
  - `Epoch 2 done`, `58.19s/epoch`, `ETA 1h54m`
  - `Epoch 3 Iter 120/227`: `Loss 9.314`, `Acc 0.034`
- 当前观察:
  1. 早期 loss 轨迹与 `exp110` 基本一致，说明 `MIN_COUNT=4` 没有影响 warmup 前主干收敛
  2. 当前还看不到 `sckd` 分项，这符合设计：`warmup=20`
  3. epoch 时长约 `58-60s`，与 `exp110` 基本相同，没有额外开销
- 当前判断: 继续
- 原因:
  - 这次改动只影响 `epoch > 20` 后的蒸馏触发条件
  - 当前已排除“因 teacher reliability 门槛提升而导致启动异常”的风险
