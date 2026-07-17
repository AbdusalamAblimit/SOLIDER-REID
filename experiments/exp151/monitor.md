# exp151 PVAT 监控

## 实验信息
- 方法: PVAT（Pose-Visibility Adversarial Training）
- 类型: 表示学习范式（与 PCVT 的数据增强范式不同）
- 主基线: `exp030a-eq`（3-seed mean: 60.73% mAP / 72.57% R1）
- 运行位置: 远程 5060 Ti
- 当前状态: 准备启动

## 核心机制
- gradient reversal 强制 backbone 特征不携带 visibility 信息
- warmup (ep1-20): alpha=0, predictor 学习, backbone 不受影响
- 正式阶段 (ep21-120): alpha 线性升到 1.0, backbone 被迫隐藏 visibility

## 关键监控指标
- `pvat_acc`: 如果训练后期降向 0.5 → adversarial 在工作
- `pvat_acc` 始终很高 (>0.8) → alpha 不够或机制无效
- `pvat_loss`: 如果不收敛 → 训练不稳定

## 止损判据
1. 若 ep30 mAP 显著低于 exp030a ep30 (52.2), 说明 PVAT 在损害主学习
2. 若 pvat_acc 在 ep60 仍 > 0.9 且 mAP 无变化, 说明 adversarial 太弱但也无害
3. 若训练出现 NaN/Inf, 立即停止（gradient reversal 可能导致不稳定）

## 启动记录

### [2026-03-23 00:28] 远程启动成功，ep1 形态健康
- 远程机器: `root@i-2.gpushare.com:29162`
- 输出目录: `log/occluded_duke/exp151_pvat`
- GPU 显存: 6656 MiB（正常，单视图模型）
- 日志确认: `[PVAT] enabled: weight=0.1, warmup=20, alpha_max=1.0, vis_thr=0.5`
- ep1 早期指标:
  - `pvat_alpha = 0.000`（warmup 期，正确）
  - `pvat_acc`: 从 0.525 升到 0.769（predictor 正在学习）
  - `pvat_loss ≈ 0.69`（接近 -log(0.5)=0.693 起始值）
  - `pvat_vis_ratio ≈ 0.83`（83% 关键点可见，合理）
  - 主损失正常下降
- 当前判断:
  - 继续
  - 原因:
    1. PVAT 已正确接入
    2. predictor 在 warmup 期正常学习
    3. 下一关键点：ep10

### [2026-03-23 01:00] ep10/ep20 验证，warmup 结束，adversarial 开始
- ep10: mAP 39.1% / R1 50.4%
  - vs exp030a ep10 (38.2/51.3): +0.9/-0.9，基本持平
  - warmup 期 backbone 不受 PVAT 影响，预期结果
- ep20: mAP 47.3% / R1 60.0%
  - vs exp030a ep20 (46.8/60.9): +0.5/-0.9，基本持平
  - warmup 结束，pvat_alpha 开始从 0 升起
- ep22 日志确认: `pvat_alpha = 0.020`，gradient reversal 已生效
  - `pvat_acc = 0.831`，backbone 还未开始隐藏 visibility（alpha 太小）
- 当前判断:
  - 继续
  - 下一关键点：ep30（alpha ≈ 0.1，应能看到 pvat_acc 是否开始下降）

### [2026-03-23 01:15] ep30 验证，adversarial 刚开始生效，但已转负
- ep30: mAP 51.7% / R1 64.2%
  - vs exp030a ep30 (52.2/66.0): **-0.5/-1.8**，首次转负！
- 机制侧:
  - `pvat_alpha = 0.120`，gradient reversal 已开始
  - `pvat_acc = 0.830`，backbone 还没开始隐藏 visibility（alpha 太弱）
  - `pvat_loss = 0.444`，predictor 已收敛
- 关键判断:
  1. alpha 才 0.12，adversarial 几乎还没真正施加压力
  2. 但结果已经比 exp030a 差 -0.5 mAP / -1.8 R1
  3. 这可能是因为 pvat_loss 虽然小（0.044 × weight=0.1），仍在消耗一些优化容量
  4. 也可能只是随机波动——ep30 太早做结论
- 当前判断:
  - 继续到 ep50，观察 alpha 升高后 pvat_acc 是否开始下降
  - 如果 ep50 仍为负且 pvat_acc 未下降 → adversarial 机制无效，考虑止损

### [2026-03-23 01:28] ep40 验证，回到正向

| Epoch | PVAT | exp030a | Δ mAP | Δ R1 |
|-------|------|---------|-------|------|
| 10 | 39.1 / 50.4 | 38.2 / 51.3 | +0.9 | -0.9 |
| 20 | 47.3 / 60.0 | 46.8 / 60.9 | +0.5 | -0.9 |
| 30 | 51.7 / 64.2 | 52.2 / 66.0 | -0.5 | -1.8 |
| 40 | 56.3 / 68.8 | 55.6 / 68.6 | **+0.7** | **+0.2** |

- 观察:
  1. ep30 的负值可能只是波动，ep40 回到 +0.7
  2. `pvat_alpha=0.22`, `pvat_acc=0.829` — accuracy 未下降
  3. 说明 gradient reversal 在 alpha=0.22 时对 backbone 几乎没有影响
  4. PVAT 目前更像是一个中性附加模块
- 当前判断:
  - 继续
  - 下一关键点：ep50/60，alpha 将升到 0.3-0.4，看是否有真正效果

### [2026-03-23 01:42] ep50 验证，意外正向

| Epoch | PVAT | exp030a | Δ mAP | Δ R1 |
|-------|------|---------|-------|------|
| 10 | 39.1 / 50.4 | 38.2 / 51.3 | +0.9 | -0.9 |
| 20 | 47.3 / 60.0 | 46.8 / 60.9 | +0.5 | -0.9 |
| 30 | 51.7 / 64.2 | 52.2 / 66.0 | -0.5 | -1.8 |
| 40 | 56.3 / 68.8 | 55.6 / 68.6 | +0.7 | +0.2 |
| **50** | **57.7 / 70.3** | **55.7 / 68.8** | **+2.0** | **+1.5** |

- 观察:
  1. ep50 显示 +2.0 mAP / +1.5 R1，是目前最大正向
  2. 但 `pvat_acc = 0.835` 完全没下降，alpha=0.31 的 adversarial 几乎无效
  3. 如果不是 adversarial 造成的提升，那可能是：
     - (a) 训练方差（单 seed 波动）
     - (b) pvat_head 参数提供了轻微正则化（weight decay on extra params）
     - (c) 巧合——ep50 是 exp030a 的一个低谷点（55.7→57.7 在 ep50→60）
  4. 注意 exp030a ep50(55.7)→ep60(57.7) 有 +2.0 跳跃，可能只是 PVAT 的 ep50 碰上了 exp030a 的波谷
- 当前判断:
  - 继续，但不要过早兴奋
  - 下一关键点：ep60 将揭示 PVAT 是否真正优于 exp030a 的同期值

### [2026-03-23 01:56] ep60 验证，回到持平

| Epoch | PVAT | exp030a | Δ mAP | Δ R1 |
|-------|------|---------|-------|------|
| 40 | 56.3 / 68.8 | 55.6 / 68.6 | +0.7 | +0.2 |
| 50 | 57.7 / 70.3 | 55.7 / 68.8 | +2.0 | +1.5 |
| 60 | 57.6 / 70.2 | 57.7 / 70.8 | **-0.1** | **-0.6** |

- 观察:
  1. ep50 的 +2.0 确认只是 exp030a ep50 的波谷效应
  2. ep60 回到持平（-0.1/-0.6）
  3. `pvat_acc` 从 0.835 降到 0.821，alpha=0.42 时 adversarial 开始有微弱效果
  4. 但这个 acc 下降幅度极小，说明 backbone 特征对 visibility 的编码非常顽固
- 初步结论:
  - PVAT 目前表现为中性（与 exp030a 持平）
  - adversarial visibility removal 机制不够强来改变 backbone 行为
  - 继续到 ep120 看 alpha→1.0 时是否有变化
