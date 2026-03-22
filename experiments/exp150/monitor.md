# exp150 PCVT-Random 监控

## 实验信息
- 方法: PCVT-Random（随机块遮挡对照）
- 类型: exp148 PCVT 的关键机制对照
- 主基线: `exp030a-eq`
- 直接对照: `exp148 PCVT`（pose-guided complementary masking）
- 运行位置: 远程 5060 Ti
- 当前状态: 准备启动

## 核心机制问题
exp148 的增益来自 pose-guided complementary structure，还是仅来自三视图 + consistency 正则化？

## 硬件注意
- 远程 5060 Ti 仅 16GB VRAM，3-view PCVT 的 3x 前向需要 ~18GB → OOM
- 解决方案：启用 `WITH_CP: True`（gradient checkpointing），数学上等价，仅增加计算时间
- exp148（本地 3090 24GB）运行时 `WITH_CP: False`
- 跨硬件一致性已在多个历史实验中验证（Δ<0.4%）

## 止损判据
1. 若 ep30 显著弱于 exp030a（`mAP < 52.0`，即比 exp030a ep30 的 52.2 还差），说明随机遮挡过强
2. 若 ep30 与 exp148 几乎相同（差值 < 0.5 mAP），则需要 ep60 以上才能做出可靠判断
3. 完整 120ep 对比是最终结论的基础
