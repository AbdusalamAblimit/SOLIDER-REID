# exp112 监控

## 实验信息
- 方法: High-Confidence Support SCKD
- 类型: 训练端单变量改进
- 运行位置: 远程 5060 Ti
- 主基线: `exp110_sckd`
- 核心变量: `POSE_SCKD_UPDATE_THR = 0.7`

## 启动记录

### [2026-03-19 18:34] 实验准备
- 启动原因:
  1. `exp110` 已证明训练端 `support-complete distillation` 可转正
  2. 本地 `exp111` 正在测试 support 数量门槛（`MIN_COUNT=4`）
  3. 远程并行测试 support 纯度门槛（`UPDATE_THR=0.7`），可更快分辨 teacher reliability 的关键来源
- 当前执行内容:
  1. 保持 `exp110` 其余配置完全不变
  2. 仅提升 prototype bank 的写入可见度阈值
  3. 在远程 5060 Ti 后台启动
- 当前判断: 待启动
- 原因:
  - 这是对 `exp110` 最自然的单变量 teacher purity 验证
