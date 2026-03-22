# exp152 MaxSim Triplet 监控

## 实验信息
- 方法: Set-to-Set Metric Learning via Soft-MaxSim Triplet
- 类型: 范式级创新（vector-to-vector → set-to-set metric learning）
- 主基线: `exp030a-eq` (3-seed mean: 60.73% mAP / 72.57% R1)
- 直接对照: `exp030a + maxsim_hybrid test-only` (62.2% mAP / 74.5% R1)
- 运行位置:
  - **exp152** (soft, tau=0.05): 远程 5060 Ti
  - **exp152b** (hard, tau=0.005): 本地 3090

## 启动记录

### [2026-03-23 03:50] 远程 exp152 (soft tau=0.05) 启动
- 日志确认: `tri_maxsim` 出现，MaxSim triplet 生效
- ep1 观察:
  - `maxsim_d_ap = 0.65, maxsim_d_an = 0.50, margin = -0.15`
  - `maxsim_ent = 2.16` (log(17)=2.83, 比较分散但不 uniform)
  - margin 为负是正常的训练初期（负例距离比正例更近）

### [2026-03-23 03:54] 本地 exp152b (hard tau=0.005) 启动
- 与远程形成强消融：soft vs hard MaxSim
- ep1 观察:
  - `maxsim_ent = 0.56` (极尖锐, 接近 hard max)
  - `margin = -0.14`（与 soft 相似但收敛略慢）
  - hard MaxSim 梯度更稀疏，可能导致学习更慢

### [2026-03-23 03:55] ep3 对比

| 指标 | soft (tau=0.05) ep3 | hard (tau=0.005) ep1 |
|------|--------------------|--------------------|
| maxsim_ent | 1.95 | 0.56 |
| maxsim_margin | -0.046 | -0.135 |
| tri_maxsim | 0.717 | 0.764 |

- soft 的 margin 收敛更快（ep3 已到 -0.046 vs hard ep1 -0.135）
- 初步信号支持 soft > hard

## 止损条件
1. ep60 equal_concat mAP 低于 exp030a 同期 1.5% 以上 → 终止
2. tri_maxsim 出现 NaN → 终止
3. maxsim_margin 长期 ≤ 0 到 ep30 → 需要关注（但不一定止损）
