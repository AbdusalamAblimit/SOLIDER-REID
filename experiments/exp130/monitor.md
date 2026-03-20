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
