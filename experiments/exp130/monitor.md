# exp130 监控

## 实验信息
- 方法: Residual-KL SCRD
- 类型: 训练端单变量改进
- 运行位置: 本地 3090
- 主配置: `configs/occluded_duke/pose_psg_gcn_pair_top_residual_kl_scrd.yml`
- 核心变量: 相对 `exp125` 只新增 `POSE_CSRD_TARGET_MODE = 'residual_kl'`
- 直接对照: `exp125`
- 次对照: `exp129`

## 启动记录

### [2026-03-20 18:36] 启动前记录
- 启动原因:
  1. `exp129` 的 Claude 审查已指出其关键混淆：它实际同时改变了 target、loss family 和 normalization
  2. 而且 `base` 在 `Smooth L1` residual 写法里对 `dist_s` 的梯度方向会抵消，不能作为 “target dilution” 证据
  3. 当前本地主线应立刻改成真正隔离变量的版本，而不是继续烧到 `epoch 20+`
- 当前判断: 待启动
- 原因:
  - `exp130` 保留 `exp125` 的 KL / tau / online teacher / delta_top` 全部不变，只改 target 为 `residual_kl`

### [2026-03-20 18:38] 启动前补充
- 新规则:
  1. 按用户最新要求，所有后续实验必须先通过 Claude 审查再启动
  2. 因此 `exp130` 当前禁止直接开跑
- 当前判断: 等待审查
- 原因:
  - 先生成 `experiments/exp130/claude_review.md`，确认不存在新的变量混淆或阻塞性实现问题，再决定是否启动训练

### [2026-03-20 18:39] Claude 审查结论
- 审查文件: `experiments/exp130/claude_review.md`
- 审查结论:
  1. `exp130` 相对 `exp125` 是真正的单变量实验：
     - 保持 KL / tau / online teacher / delta_top / 主损失全不变
     - 只把 `target` 从 `full` 改成 `residual_kl`
  2. `residual_kl` 不存在 `exp129` 那种梯度退化问题，因此能作为 `target dilution` 的有效检验
  3. 无阻塞性 bug；唯一需要补的是一条保护：
     - `POSE_CSRD_TARGET_MODE=residual_kl` 也应要求 `POSE_CSRD_SUPPORT_TEACHER=True`
  4. 主要实验风险不是实现错误，而是：
     - residual logits 量级更小，`tau=0.10` 可能让 teacher 分布更软
- 已处理:
  1. 已在 `processor.py` 补上 `residual_kl requires support teacher` 的保护
- 当前判断: 允许启动
- 原因:
  - 审查明确支持启动；剩余问题属于方法风险，不是实现阻塞

### [2026-03-20 18:40] 启动确认（本地 3090）
- 运行位置: 本地 3090
- 输出目录: `log/occluded_duke/exp130_pair_top_residual_kl_scrd`
- 配置: `configs/occluded_duke/pose_psg_gcn_pair_top_residual_kl_scrd.yml`
- 关键确认:
  1. 主训练进程已成功启动：
     - `python -u train.py --config_file configs/occluded_duke/pose_psg_gcn_pair_top_residual_kl_scrd.yml`
  2. support-complete teacher 仍为在线版本：
     - `[CSRD-ST] enabled: ... stop_epoch=-1`
  3. 新 target 已正确接线：
     - `[CSRD-TARGET] mode=residual_kl`
  4. `delta_top` pair routing 仍保持不变：
     - `[CSRD-PW] mode=delta_top, alpha=1.0, top_ratio=0.25`
- 当前判断: 继续
- 原因:
  - `exp130` 已经通过“先审查、再启动”的新规则；下一关键点是 warmup 稳定性与 `epoch 21+` 的 `csrd` 量级

### [2026-03-20 18:33] 检查点 #1 — Epoch 10
- 结果:
  - `ep10 = 39.8% / 52.9% / 68.8% / 74.3%`
- 对照:
  - `exp125 ep10 = 38.3 / 51.4`
  - `exp119 ep10 = 39.8 / 52.9`
  - `exp120 ep10 = 39.8 / 52.9`
- 当前观察:
  1. warmup 前段没有被 `residual_kl` 改动拖坏
  2. 早期验证直接贴近 `exp119/120`，也略高于 `exp125` 的 warmup 点
  3. 当前仍不能据此判断方法正负，因为 `CSRD` 尚未激活
- 当前判断: 继续
- 原因:
  - 真正关键的仍是 `epoch 21+` 后 `csrd` 分项有没有明显偏小，以及 late-stage 指标能否接住

### [2026-03-20 20:38] 检查点 #2 — Epoch 80 / 90
- 结果:
  - `ep80 = 59.0% / 71.5% / 83.7% / 87.4%`
  - `ep90 = 59.7% / 73.1% / 84.4% / 88.1%`
- 对照:
  - `exp125 ep80 = 59.4 / 72.0`
  - `exp125 ep90 = 60.1 / 73.9`
  - `exp030a ep90 = 59.4 / 72.6`
- `CSRD` 统计（epoch 78-92）:
  - `csrd = 0.011~0.013`
  - `csrd_tgap = 0.558~0.568`
  - `csrd_sgap = 0.560~0.575`
  - `csrd_pd = 0.002~0.003`
  - `csrd_pf = 1.11~1.15`
  - `csrd_psr = 0.89~0.92`
  - `csrd_tr ≈ 0.001`
- 当前观察:
  1. `residual_kl` 没有出现“`csrd` 接近 0”这种失效现象，说明审查里担心的 `tau` 过软目前**不是致命问题**
  2. 但到 `ep80/90` 为止，它仍稳定落后于 `exp125`：
     - `ep80`: `-0.4 / -0.5`
     - `ep90`: `-0.4 / -0.8`
  3. 相对 `exp030a` 仍是弱正向，因此这条线不是负方向；只是当前**不如 full-target 的 exp125**
  4. 这会把判断收紧成：
     - `target dilution` 也许存在，但它目前不像主瓶颈
     - 至少在 `delta_top + online teacher` 这套框架下，完整 teacher distribution 仍更有效
- 当前判断: 继续到 `ep100` 再做最终 late-stage 决策
- 原因:
  - 实验已进入后期，继续多跑一个关键验证点成本低；若 `ep100` 仍落后 `exp125`，就可较有把握地结束这条 `residual target` 线

### [2026-03-20 20:56] 检查点 #3 — Epoch 100
- 结果:
  - `ep100 = 59.7% / 73.3% / 84.4% / 87.9%`
- 对照:
  - `exp125 ep100 = 60.0 / 73.1`
  - `exp125 ep110 = 60.4 / 73.8`
  - `exp030a ep90 = 59.4 / 72.6`
- `CSRD` 统计（epoch 100 前后）:
  - `csrd = 0.011~0.013`
  - `csrd_tgap = 0.570~0.572`
  - `csrd_sgap = 0.575~0.578`
  - `csrd_pd = 0.002~0.003`
  - `csrd_pf = 1.13~1.14`
  - `csrd_psr = 0.90~0.91`
  - `csrd_tr ≈ 0.001`
- 当前观察:
  1. 到 `ep100` 为止，`residual_kl` 不再是单纯落后，而是变成：
     - 相对 `exp125 ep100`: `mAP -0.3 / R1 +0.2`
  2. 这说明它至少不是明显失败；和 `exp125` 的差距已经收窄成 late-stage trade-off
  3. 同时 `csrd` 仍维持在 `0.012` 左右，明显高于 `exp125 ep100` 的 `0.009`，再次说明：
     - residual target 没有因为 `tau` 过软而失去训练信号
  4. 但它当前仍没有给出 “target dilution 是主瓶颈” 的强证据，因为：
     - `mAP` 还没反超 `exp125`
     - `exp125` 自身在 `ep110` 还有一次明显抬升
- 当前判断: 继续到 `ep110`
- 原因:
  - 现在直接停表还太早；若 `exp130` 能在 `ep110` 延续这种收敛趋势，才有资格和 `exp125` 做真正 late-stage 对比

### [2026-03-20 20:58] 检查点 #4 — Epoch 105 前后训练态
- 当前进度:
  - 已完成 `Epoch 105`
  - 最近一次验证仍为 `ep100 = 59.7% / 73.3% / 84.4% / 87.9%`
- `CSRD` 统计（epoch 101-105）:
  - `csrd = 0.011~0.013`
  - `csrd_tgap = 0.565~0.572`
  - `csrd_sgap = 0.573~0.578`
  - `csrd_pd = 0.002~0.003`
  - `csrd_pf = 1.11~1.15`
  - `csrd_psr = 0.89~0.92`
  - `csrd_sr = 0.14~0.15`
  - `csrd_sn = 153~166`
- 当前观察:
  1. `exp130` 在 `ep100` 之后没有出现发散或塌缩，主损失、`csrd`、`student_gap` 都保持稳定
  2. `csrd` 信号仍然足够强，说明 `residual_kl` 不是“接上了但几乎没学”的假阳性
  3. 但从当前训练态看，也没有出现能支持“后面会明显反超 `exp125`”的强信号；它更像继续维持 `mAP` 略低、`R1` 接近或略高的 trade-off
  4. 因而当前最合理的结论仍是：
     - `target dilution` 不是当前最强瓶颈候选
     - `residual target` 线可以继续观察，但不应再上升为下一阶段主创新假设
- 当前判断: 继续到 `ep110`
- 原因:
  - 训练已经进入最后一段，继续看到 `ep110` 的成本很低；若 `ep110` 仍未压过 `exp125`，即可较干净地结束这条 `residual target` 支线

### [2026-03-20 21:19] 检查点 #5 — Epoch 110 / 120（最终）
- 结果:
  - `ep110 = 60.1% / 73.4% / 84.5% / 88.3%`
  - `ep120 = 60.1% / 73.1% / 84.6% / 88.3%`
- 对照:
  - `exp125 ep110 = 60.4 / 73.8`
  - `exp125 ep120 = 60.5 / 73.5`
  - `exp030a-eq seed1234 = 61.1 / 72.9`（正式 eval）
- `CSRD` 统计（epoch 110-120）:
  - `csrd = 0.011~0.013`
  - `csrd_tgap = 0.567~0.579`
  - `csrd_sgap = 0.574~0.584`
  - `csrd_pd = 0.001~0.002`
  - `csrd_pf = 1.12~1.14`
  - `csrd_psr = 0.90~0.91`
  - `csrd_sr = 0.14~0.15`
  - `csrd_sn = 152~169`
- 当前观察:
  1. `residual_kl` 到收敛都没有失效，`csrd` 量级和 gap 统计始终稳定，说明它不是“信号太弱所以看起来没效果”
  2. 但最终 `ep110/120` 都稳定落后于 `exp125`：
     - 相对 `ep110`: `-0.3 / -0.4`
     - 相对 `ep120`: `-0.4 / -0.4`
  3. 因而当前可以更有把握地下结论：
     - `target dilution` 不是当前主瓶颈
     - 至少在 `delta_top + online support teacher` 这条线上，完整 teacher target 比 `residual_kl` 更有效
  4. 这条线的价值主要变成“负向因果证据”：
     - 说明下一步不该继续改 `target form`
     - 而应回到 **pair coverage / pair selection** 本身
- 当前判断: 结束该线
- 原因:
  - `ep110` 与 `ep120` 已经足够把 `residual target` 判成次优支线，继续延伸价值很低

### [2026-03-20 18:40] Claude 审查结论
- 审查文件: `experiments/exp130/claude_review.md`
- 审查结论:
  1. `exp130` 相对 `exp125` 是真正的单变量实验：
     - 保持 KL / tau / online teacher / delta_top 全不变
     - 只把 `target` 从 `full` 改成 `residual_kl`
  2. `residual_kl` 不存在 `exp129` 那种梯度退化问题，因此可以作为 `target dilution` 的有效检验
  3. 无阻塞性 bug，但有一条中优先级保护缺失：
     - `residual_kl` 模式也应要求 `POSE_CSRD_SUPPORT_TEACHER=True`
  4. 另一个主要风险不是实现错误，而是：
     - residual logits 量级可能更小，`tau=0.10` 可能让 teacher 分布变得更软
- 已处理:
  1. 已在 `processor.py` 中补上 `residual_kl requires support teacher` 的保护
- 当前判断: 允许启动
- 原因:
  - 审查已明确支持 `exp130`；当前剩余风险属于实验假设风险，不是实现阻塞
