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
