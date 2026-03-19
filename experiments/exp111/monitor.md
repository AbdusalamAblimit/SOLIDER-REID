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
