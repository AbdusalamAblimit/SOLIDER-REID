# exp152 MaxSim Triplet 监控

## 实验信息
- 方法: Set-to-Set Metric Learning via Soft-MaxSim Triplet
- 类型: 范式级创新（从 vector-to-vector 到 set-to-set metric learning）
- 主基线: `exp030a-eq` (3-seed mean: 60.73% mAP / 72.57% R1)
- 直接对照: `exp030a + maxsim_hybrid test-only` (62.2% mAP / 74.5% R1)
- 当前状态: 准备启动

## 核心假设
MaxSim training + MaxSim test > MaxSim test-only > pooled vector baseline

## 关键监控指标
- `tri_maxsim`: MaxSim triplet loss
- `maxsim_d_ap / maxsim_d_an / maxsim_margin`: 距离分布
- `maxsim_ent`: 注意力熵（低=hard max行为，高=uniform）

## 止损条件
1. ep60 equal_concat mAP 低于 exp030a 同期 1.5% 以上 → 终止
2. tri_maxsim 出现 NaN → 终止
3. maxsim_margin 长期 ≤ 0 → 机制失效
