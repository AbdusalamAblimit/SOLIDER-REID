# exp131 监控

## 实验信息
- 方法: Cross-Batch Pair SCRD
- 类型: 训练端单变量改进
- 运行位置: 本地 3090
- 主配置: `configs/occluded_duke/pose_psg_gcn_pair_queue_scrd.yml`
- 核心变量: 相对 `exp125` 只新增 `POSE_CSRD_QUEUE_SIZE = 256`
- 直接对照: `exp125`
- 次对照: `exp126`

## 启动记录

### [2026-03-20 21:25] 启动前记录
- 启动原因:
  1. `exp130` 已较干净地否定“target 改写就是主突破口”
  2. `exp125` 已证明 pair routing 有效，但 batch-only changed pairs 可能覆盖不足
  3. 因而本地下一步不再改 target，而改 relation coverage
- 当前判断: 待审查
- 原因:
  - 按最新规则，必须先完成 Claude 审查，再允许启动训练

### [2026-03-20 23:24] Claude 首轮审查结论
- 审查文件: `experiments/exp131/claude_review.md`
- 审查结论:
  1. `exp131` 相对 `exp125` 的 config diff 是干净的，仍是单变量
  2. 但存在一个阻塞性 bug：
     - `processor.py` 中 enqueue 阶段错误地从 `kp_data` 读取 `csrd_teacher_feats`
     - 该 key 只存在于 `kp_aux_data`
     - 导致 queue 永远为空，实验会静默退化成 `exp125`
- 已处理:
  1. 已将 enqueue 阶段的读取对象修正为 `kp_aux_data.get('csrd_teacher_feats')`
  2. 已通过 `py_compile` 复查
- 当前判断: 待二次审查
- 原因:
  - 按规则，修完阻塞性问题后仍需重新审查，确认现在已经允许启动

### [2026-03-20 23:31] Claude 二次审查结论
- 审查文件:
  - 首轮: `experiments/exp131/claude_review.md`
  - 二次: `experiments/exp131/claude_review_v2.md`
- 审查结论:
  1. 首轮指出的 critical bug 已被正确修复
  2. `exp131` 相对 `exp125` 仍然是单变量：
     - 唯一新增 `POSE_CSRD_QUEUE_SIZE = 256`
     - teacher / target / routing / alpha / top_ratio 全保持不变
  3. 无新的阻塞性问题，允许启动
  4. 主要方法风险是：
     - queue relation 进入 softmax 后可能稀释单 pair 信号
     - stale queue features 可能引入一定噪声
- 当前判断: 允许启动
- 原因:
  - 当前已经满足“先审查、后启动”的规则，可以正式启动训练验证 coverage 假设

### [2026-03-20 23:30] 启动确认（本地 3090）
- 运行位置: 本地 3090
- 输出目录: `log/occluded_duke/exp131_pair_queue_scrd`
- 配置: `configs/occluded_duke/pose_psg_gcn_pair_queue_scrd.yml`
- 关键确认:
  1. queue 开关已正确生效：
     - `[CSRD-QUEUE] size=256`
  2. 其余主线参数保持 `exp125` 不变：
     - `[CSRD-TARGET] mode=full`
     - `[CSRD-PW] mode=delta_top, alpha=1.0, top_ratio=0.25`
     - `[CSRD-ST] enabled: ... stop_epoch=-1`
  3. 数据、模型、优化器初始化全部正常，无启动期报错
- 当前判断: 继续
- 原因:
  - 当前已经完成最关键的接线确认；下一步只需要验证 warmup 稳定性与 queue 统计是否在 `epoch 21+` 正常出现

### [2026-03-20 23:31] 检查点 #1 — Epoch 1 warmup 前段
- 当前进度:
  - `Epoch[1] Iter[140/227]`
- 当前局部训练状态:
  - `Loss: 16.498`
  - `Acc: 0.001`
  - `id_global: 6.554`
  - `id_part: 6.641`
  - `tri_global: 9.024`
  - `tri_part: 10.776`
- 当前观察:
  1. warmup 前段形状与 `exp125` 的正常起步一致，没有因为 queue 机制造成初始化异常
  2. 目前还未进入 `CSRD` 激活区，因此现在能确认的只有：
     - queue 机制没有污染 warmup
     - 主训练路径保持稳定
  3. 这也说明第一次 `nohup` 留驻失败更像启动方式问题，而不是代码问题
- 当前判断: 继续
- 原因:
  - 真正关键的监控点是 `ep10 / ep20`，以及 `epoch 21+` 后 `csrd_qn / csrd_qr` 是否表明 cross-batch relations 真正参与了 distillation

### [2026-03-21 02:23] 检查点 #2 — Epoch 110 / 120（最终）
- 结果:
  - `ep110 = 60.4% / 73.7% / 84.9% / 87.8%`
  - `ep120 = 60.5% / 73.7% / 84.8% / 88.0%`
- 对照:
  - `exp125 ep110 = 60.4 / 73.8`
  - `exp125 ep120 = 60.5 / 73.5`
  - `exp130 ep120 = 60.1 / 73.1`
- `CSRD` 统计（epoch 110-120）:
  - `csrd = 0.008`
  - `csrd_pf = 1.11~1.13`
  - `csrd_psr = 0.90~0.91`
  - `csrd_sr = 0.14~0.15`
  - `csrd_sn = 152~169`
  - `csrd_qn = 256`
  - `csrd_qr = 0.427~0.441`
- 当前观察:
  1. queue 不是“接上了但没参与”的假阳性；后期始终有约 `43%` 的候选 relations 来自 cross-batch queue，coverage 扩展是真实发生的
  2. 但最终结果相对 `exp125` 只体现为：
     - `mAP +0.0`
     - `R1 +0.2`
     这不足以支撑“batch 内 changed-pair coverage 不足”是当前主瓶颈
  3. 同时 `pair_focus / pair_select_ratio` 几乎没有比 `exp125` 更强，说明把更多 candidate pairs 喂给当前 student，并没有自动转成更有用的 pair correction
  4. 因而当前更合理的解释是：
     - changed pairs 并不稀缺到需要 queue 才能看见
     - 真正卡住的更像是 **pair-specific support-complete correction 不能被当前单向量学生充分吸收**
- 当前判断: 结束该线
- 原因:
  - `exp131` 已足够回答 coverage 问题；继续扫 queue 大小或 stale 策略只会回到低价值局部调参
