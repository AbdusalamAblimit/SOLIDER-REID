# exp116 监控

## 实验信息
- 方法: Support-Complete Feature Replacement (SCFR)
- 类型: 训练端单变量改进
- 运行位置: 本地 3090
- 主配置: `configs/occluded_duke/pose_psg_gcn_scfr.yml`
- 核心变量: `POSE_SCFR = True`（直接替换低可见度 keypoint 特征，而非蒸馏 loss）
- 对照组: `exp110_sckd`（cosine distillation 版）

## 启动记录

### [2026-03-20 04:46] 启动确认
- 运行位置: 本地 3090
- 输出目录: `log/occluded_duke/exp116_scfr`
- 关键确认:
  1. `[SCKD] enabled: weight=0.5, warmup=20, low_thr=0.3, update_thr=0.5, mom=0.9, stop_epoch=-1`
  2. `[SCFR] Feature replacement mode enabled (loss disabled, bank replaces features)`
  3. 训练正常启动，无报错
  4. SCKD loss 不出现在日志中（SCFR 模式下跳过）
- 当前判断: 继续
- 原因:
  - agent review 已通过（两轮）
  - 配置与 exp110 仅差 `POSE_SCFR=True`
  - 这是第一个测试"直接替换 vs 蒸馏 loss"的实验

### [2026-03-20 05:10] 检查点 #1 — Epoch 10-21

- 结果:
  - `ep10 = 38.3% / 51.3%`（与 exp110 完全一致，warmup 阶段无差异）
  - `ep20 = 47.1% / 59.7%`（与 exp110 完全一致，warmup 结束）
- SCFR 激活确认（ep21 Iter20）：
  - `scfr_n = 157.350`（每 batch 约 157 个 keypoint 被替换）
  - `scfr_r = 0.145`（14.5% 替换率，约 2.5 kp/sample out of 17）
  - 无 SCKD loss 出现（正确跳过）
- 当前观察:
  1. warmup 阶段与所有 SCKD 变体完全一致，符合预期
  2. SCFR 在 ep21 开始正确激活，替换率与 SCKD 的 low_ratio 一致
  3. 关键观察点：ep30/40 是否出现与 SCKD 的分化
- 当前判断: 继续
- 原因:
  - SCFR 正确运行，需要等 ep30/40 看是否有差异

### [2026-03-20 05:20] 检查点 #2 — Epoch 30

- 结果:
  - `ep30 = 52.6% / 65.2% / 79.5% / 84.3%`
- 对照:
  - `exp110 ep30` = `52.6 / 65.4` → `0.0 / -0.2`
  - `exp114 ep30` = `52.6 / 65.2` → `0.0 / 0.0`
- SCFR 统计（ep30 附近）：
  - `scfr_n ≈ 157`
  - `scfr_r ≈ 0.145`
- 当前观察:
  1. ep30 与所有 SCKD 变体完全一致
  2. SCFR 已活跃 10 个 epoch，但尚未形成可见差异
  3. 这并不意外：SCFR 不添加 loss，只改变 GCN 输入
  4. 效果可能需要更多 epoch 积累
- 当前判断: 继续
- 原因:
  - 需要 ep40/50/60 来判断 SCFR 是否有实质影响

### [2026-03-20 05:30] 检查点 #3 — Epoch 40

- 结果:
  - `ep40 = 56.0% / 67.7% / 81.7% / 85.4%`
- 对照:
  - `exp110 ep40` = `56.2 / 68.4` → `-0.2 / -0.7`
  - `exp114 ep40` = `56.3 / 68.6` → `-0.3 / -0.9`
- SCFR 统计（ep37 附近）：
  - `scfr_n ≈ 160`
  - `scfr_r ≈ 0.147`
- 当前观察:
  1. **SCFR 在 ep40 略低于 SCKD 变体**（-0.2 mAP, -0.7 R1）
  2. R1 的差距更明显，说明 SCFR 可能对 top-1 精度有轻微负面影响
  3. 但差距不大，可能是早期波动。exp114 在 ep50 也曾落后再追上
  4. 如果到 ep60 仍持续落后，则 SCFR 很可能是中性偏负
- 当前判断: 继续
- 原因:
  - 还需要 ep50/60 来确认趋势

### [2026-03-20 05:42] 检查点 #4 — Epoch 50

- 结果:
  - `ep50 = 56.1% / 68.6% / 81.0% / 84.7%`
- 对照:
  - `exp110 ep50` = `56.1 / 68.3` → `0.0 / +0.3`
  - `exp114 ep50` = `56.2 / 68.5` → `-0.1 / +0.1`
- 当前观察:
  1. **SCFR 在 ep50 回到了与 SCKD 完全等价的水平**
  2. ep40 的 -0.2 mAP 差距已消失
  3. SCFR（直接替换）与 SCKD（蒸馏 loss）到 ep50 给出相同结果
  4. 需要看 ep60/70/80 才能判断后期是否分化
- 当前判断: 继续
- 原因:
  - 到目前为止无显著差异，需要更多数据
