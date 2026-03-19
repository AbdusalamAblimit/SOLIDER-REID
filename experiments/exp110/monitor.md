# exp110 SCKD 监控

## 实验信息
- 方法: SCKD（Support-Complete Keypoint Distillation）
- 类型: 训练端最小原型
- 主基线: `exp030a-eq`
- 核心变量: per-ID / per-keypoint prototype bank 蒸馏 low-visibility keypoints

## 启动记录

### [2026-03-19 16:25] 实验启动
- 来自 `exp109` 的直接证据：
  - `oracle_feat_only_cvk = 66.15 / 77.87`
  - support-complete 方向存在强 headroom
- 当前执行内容：
  1. 新增 `SupportCompleteBank`
  2. 在 `processor` 中接入训练期 prototype distillation
  3. 使用 `exp030a` 主配置，仅改 `SCKD`
  4. 若早期曲线明显落后，先检查 bank 覆盖率与 loss 量级，再决定是否继续

### [2026-03-19 16:39] 检查点 #1 — Epoch 1-3
- 训练状态: 正常
- 关键日志:
  - `Epoch 1 done`, `59.86s/epoch`, `ETA 1h58m`
  - `Epoch 2 done`, `58.12s/epoch`, `ETA 1h54m`
  - `Epoch 3 Iter 160/227`: `Loss 9.162`, `Acc 0.041`
- 当前观察:
  1. loss 轨迹与 `exp030a` 正常收敛形状一致，没有因 prototype bank 引入异常震荡
  2. 当前还看不到 `sckd` 分项，因为正式配置 `warmup=20`，前 20 个 epoch 仍是纯基线训练 + bank 后台积累
  3. 速度约 `58-60s/epoch`，相对 baseline 没有明显额外开销
- 当前判断: 继续
- 原因:
  - 代码路径已通过 smoke test
  - 正式训练前 3 个 epoch 没有出现 NaN / OOM / 明显退化
  - 真正的方向判断要等 `epoch > 20` 后 `SCKD` loss 激活

### [2026-03-19 16:48] 检查点 #2 — Epoch 10 验证
- 结果:
  - `mAP 38.3% / R1 51.3% / R5 66.7% / R10 73.3%`
- 对照:
  - `exp030a` 在 `ep10` 约为 `38.2% / 51.3%`
- 当前观察:
  1. `exp110` 在 warmup 阶段与 baseline 几乎完全重合，说明 prototype bank 的后台更新没有破坏主干优化
  2. 这一步还不能说明 `SCKD` 有效，因为蒸馏损失尚未激活
  3. 但至少排除了“为了维护 support bank，前 10 个 epoch 就明显掉点”的风险
- 当前判断: 继续
- 原因:
  - 训练曲线和早期验证都与 baseline 对齐
  - 下一关键观察点是 `epoch 20`（warmup 结束）及其后的 `sckd` 分项是否稳定出现
