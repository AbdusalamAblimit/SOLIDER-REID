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
3. maxsim_margin 长期 ≤ 0 到 ep30 → 需要关注

### [2026-03-23 05:40] ep10-100 结果汇总 — MaxSim triplet 呈现严重负向 🔴

| Epoch | exp030a | soft (远程) | Δ soft | hard (本地) | Δ hard |
|-------|---------|------------|--------|------------|--------|
| 10 | 38.2 / 51.3 | 37.7 / 51.7 | -0.5 | 38.4 / 52.4 | +0.2 |
| 20 | 46.8 / 60.9 | 47.8 / 61.5 | +1.0 | 47.4 / 61.2 | +0.6 |
| 30 | 52.2 / 66.0 | 52.7 / 65.2 | +0.5 | 53.4 / 66.4 | +1.2 |
| 40 | 55.6 / 68.6 | 56.0 / 68.3 | +0.4 | 55.9 / 68.1 | +0.3 |
| 50 | 55.7 / 68.8 | 55.8 / 68.9 | +0.1 | 54.9 / 67.6 | -0.8 |
| 60 | 57.7 / 70.8 | 55.2 / 67.4 | **-2.5** | 55.1 / 66.7 | **-2.6** |
| 70 | 58.1 / 70.9 | 54.8 / 67.7 | **-3.3** | 55.0 / 66.9 | **-3.1** |
| 80 | 59.4 / 71.6 | — | — | 56.3 / 67.7 | **-3.1** |
| 90 | 59.8 / 72.1 | — | — | 57.3 / 69.0 | **-2.5** |
| 100 | 60.2 / 73.0 | — | — | 57.2 / 69.3 | **-3.0** |

- 分析:
  1. 两条线在 ep30 前正向或持平，ep50 后急剧恶化
  2. Soft 和 Hard 表现几乎相同，说明问题不在 tau 选择
  3. **根本原因**: MaxSim triplet 梯度与 pooled ID loss 梯度冲突
     - ID loss 要求 pooled feature 有判别力 → 所有 kp 特征趋同
     - MaxSim triplet 要求每个 kp 独立判别 → kp 特征趋异
     - 两者在 ep50 后产生冲突，导致 GCN branch 特征质量下降
  4. **重要**: 以上是 equal_concat 评估。需要用 MaxSim test 重新评估这些 checkpoint
  5. 但 -3% 的 equal_concat 退化太大，即使 MaxSim test 能回补也难以超越 test-only 的 62.2%

- 当前判断: 让两台继续跑到 ep120 收集完整证据，但 MaxSim triplet 替换策略基本失败

### [2026-03-23 06:00] exp152b (hard) 完成 — MaxSim training 确认失败 🔴

**最终结果 (ep120):**

| 评估模式 | exp152b (hard MaxSim training) | exp030a (baseline) | Δ |
|----------|-------------------------------|-------------------|---|
| equal_concat | 57.8% / 69.7% | 61.1% / 73.7% | **-3.3 / -4.0** |
| maxsim_hybrid 1:2 | 59.0% / 71.0% | 62.2% / 74.5% | **-3.2 / -3.5** |

**结论：**
1. MaxSim training 严重损害了特征质量（-3.3% equal_concat）
2. 即使用 MaxSim test-time 评估也不能回补（-3.2% vs baseline MaxSim test-only）
3. "train-test metric alignment" 假设在 "替换 pooled triplet" 实现下不成立
4. 根本原因：MaxSim triplet 让每个 keypoint 独立优化 set-matching，但 pooled ID loss 要求所有 keypoint 协同做分类。两者梯度冲突导致 GCN branch 特征质量下降
5. MaxSim 应定位为 **纯 test-time method**，不适合替换训练目标
