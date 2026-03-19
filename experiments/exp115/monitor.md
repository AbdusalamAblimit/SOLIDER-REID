# exp115 监控

## 实验信息
- 方法: Freeze-Later Reliable SCKD
- 类型: 训练端单变量改进
- 运行位置: 远程 5060 Ti
- 主配置: `exp112_sckd_up07`
- 核心变量: `POSE_SCKD_UPDATE_STOP_EPOCH = 30`

## 启动记录

### [2026-03-19 19:05] 实验准备
- 启动原因:
  1. 远程当前空闲，可用于并行验证 `teacher stability`
  2. `exp114` 测的是最强冻结版本 `freeze20`
  3. 还需要一个互补对照，区分 “冻结本身有害” 和 “冻结时机过早”
- 当前执行内容:
  1. 保持 `update_thr=0.7`
  2. 仅把 `POSE_SCKD_UPDATE_STOP_EPOCH` 设为 `30`
  3. 在远程 5060 Ti 后台启动
- 当前判断: 待启动
- 原因:
  - 这是与 `exp114` 最互补、最省时间的并行单变量

### [2026-03-20 03:01] 启动确认（远程 5060 Ti）
- 运行位置: 恒源云 5060 Ti
- 远程仓库: 已同步到 `94fd7d1`
- 启动方式: 后台 `nohup`
- 输出目录: `log/occluded_duke/exp115_sckd_up07_freeze30`
- nohup 日志: `log/occluded_duke/exp115_sckd_up07_freeze30/remote_nohup.log`
- 关键确认:
  1. 配置已生效：`update_thr=0.7, stop_epoch=30`
  2. 日志已打印：
     - `[SCKD] enabled: weight=0.5, warmup=20, low_thr=0.3, update_thr=0.7, mom=0.9, stop_epoch=30`
  3. GPU 已占用约 `6.7GB`
  4. `Epoch[1] Iter[60/227] Loss: 19.126`
- 当前判断: 继续
- 原因:
  - 现在形成了本地 `freeze20` + 远程 `freeze30` 的并行对照

### [2026-03-20 03:37] 检查点 #1 — Epoch 20-23

- 结果:
  - `ep20 = 47.8% / 61.5% / 75.7% / 80.7%`
- 对照:
  - `exp110 ep20` ≈ `47.1 / 59.7`
  - `exp112 ep20` = `47.8 / 61.5`
- SCKD 机制统计（ep21 Iter20，刚开始 SCKD）：
  - `sckd = 0.165`
  - `sckd_cos = 0.836`
  - `sckd_count = 292.9`
  - `sckd_pairs = 157.4`
- SCKD 机制统计（ep23 Iter160）：
  - `sckd = 0.168`
  - `sckd_cos = 0.833`
  - `sckd_count = 349.5`
  - 注意：bank 仍在更新中（stop_epoch=30），count 在增长
- 当前观察:
  1. `ep20` 评估与 `exp112` 一致（47.8/61.5），与 `exp110` 相比 +0.7/+1.8，说明 `UPDATE_THR=0.7` 的早期正信号是一致的
  2. bank 仍在更新中，`sckd_cos` 已从 0.836 降到 0.833，符合在线 teacher 变硬的预期
  3. count 在 ep21-23 已从 ~293 增长到 ~350，说明在线更新正在积极工作
  4. 到 ep30 后 bank 将冻结，届时可以和 exp114（freeze20）做直接对比
  5. 远程速度约 ~90s/epoch vs 本地 ~55s/epoch，进度滞后约 1.6x
- 当前判断: 继续
- 原因:
  - 仍在 warmup/早期阶段，需要等到 ep40/50 才能开始有意义的对比
  - 关键是看 ep30 冻结后的走势

### [2026-03-20 03:45] 检查点 #2 — Epoch 27

- 当前进度: Epoch 27/120
- SCKD 机制统计（ep27 Iter140）：
  - `sckd = 0.183`
  - `sckd_cos = 0.818`
  - `sckd_count = 404.6`（仍在增长，stop_epoch=30 尚未触发）
  - `sckd_conf = 0.882`
  - `sckd_pairs = 155.0`
- 当前观察:
  1. bank 仍在在线更新中，count 从 ep21 的 ~293 增长到 ~405
  2. `sckd_cos` 已从 ep21 的 0.836 降到 0.818，符合在线 teacher 变硬的规律
  3. 还有 3 个 epoch（ep28-30）的在线更新，之后 bank 将冻结
  4. 值得关注：ep30 冻结时 bank 将有约 10 个 epoch 的额外积累，count 可能达到 ~500+
     - vs exp114 freeze20 时的 count ~280
     - 这意味着 exp115 的 frozen teacher 更"成熟"但也更"硬"
  5. 远程速度约 91s/epoch，预计 ep30 eval 在约 30 分钟后出现
- 当前判断: 继续
- 原因:
  - 仍处于早期，需等 ep40/50 才能开始有意义的比较

### [2026-03-20 03:55] 检查点 #3 — Epoch 30-32

- 结果:
  - `ep30 = 53.6% / 66.4% / 80.3% / 85.2%`
- 对照:
  - `exp110 ep30` = `52.6 / 65.4`
  - `exp112 ep30` = `53.6 / 66.4`（完全一致）
  - `exp114 ep30` = `52.6 / 65.2`
- SCKD 机制统计（ep32 Iter80）：
  - `sckd = 0.198`
  - `sckd_cos = 0.803`
  - `sckd_count = 431.9`（bank 刚在 ep30 冻结，count 不再增长）
  - `sckd_conf = 0.883`
- 当前观察:
  1. `ep30` 与 `exp112` 完全一致，符合预期（两者到 ep30 的 bank 状态几乎相同）
  2. bank 已在 ep30 冻结，从 ep31 起 count 应该稳定在 ~430 附近
  3. 与 exp114 的对比：
     - exp115 freeze30: count=430, cos=0.803
     - exp114 freeze20: count=300, cos=0.810（当时）
     - exp115 的 teacher 更"成熟"（更多 support）但更"硬"（cos 更低）
  4. **关键问题**：freeze30 是否能利用更成熟的 teacher 在后期超过 freeze20？
  5. 远程速度约 ~91s/epoch，ep40 eval 预计约 20 分钟后
- 当前判断: 继续
- 原因:
  - ep30 只是 bank 刚冻结的起点，真正有意义的对比要从 ep50 开始

### [2026-03-20 04:22] 检查点 #4 — Epoch 50

- 结果:
  - `ep50 = 57.5% / 69.5% / 82.5% / 86.5%`
- 对照:
  - `exp110 ep50` = `56.1 / 68.3` → `+1.4 / +1.2`
  - `exp112 ep50` = `57.4 / 69.7` → `+0.1 / -0.2`
  - `exp114 ep50` = `56.2 / 68.5` → `+1.3 / +1.0`
- SCKD 机制统计（ep50 Iter200）：
  - `sckd = 0.194`
  - `sckd_cos = 0.807`
  - `sckd_count = 507.9`（frozen 状态，不增长）
- 当前观察:
  1. **exp115 freeze30 在 ep50 显著优于 exp114 freeze20（+1.3 mAP）和 exp110 online（+1.4 mAP）**
  2. 与 exp112（online thr=0.7）几乎一致，这说明：
     - freeze30 继承了 exp112 在 ep30 前的优势
     - ep30→50 之间，冻结 bank 没有造成明显损失
  3. 关键问题：exp114 在 ep50 也曾落后但后来追上。exp115 的 ep50 优势能否持续？
  4. cos=0.807 vs exp114 同期 ~0.806，两者接近
  5. count=508 vs exp114 的 ~300，exp115 的 teacher 更成熟
- 当前判断: 继续
- 原因:
  - ep50 的领先可能是"更成熟的 frozen teacher"的早期优势
  - 需要看 ep70/80 来判断这个优势是否能持续

### [2026-03-20 04:35] 检查点 #5 — Epoch 60

- 结果:
  - `ep60 = 57.8% / 70.0% / 82.6% / 86.7%`
- 对照:
  - `exp110 ep60` = `58.3 / 70.5` → `-0.5 / -0.5`
  - `exp112 ep60` = `57.7 / 70.0` → `+0.1 / 0.0`
  - `exp114 ep60` = `58.4 / 70.5` → `-0.6 / -0.5`
- SCKD 机制统计（ep60 附近）：
  - `sckd_cos = 0.812`
  - `sckd_count = ~409`（frozen，稳定）
- 当前观察:
  1. freeze30 基本与 exp112（online thr=0.7）等价
  2. 但低于 exp114（freeze20）约 0.6 mAP
  3. 这个模式和 exp114 的 ep50-60 演化相反——exp114 在 ep50 落后但 ep60 反超
  4. 需要看后续是否 exp115 也有类似的后期追赶趋势
- 当前判断: 继续
- 原因:
  - 远程仍有约 1 小时训练时间，需要看 ep80/90/120
