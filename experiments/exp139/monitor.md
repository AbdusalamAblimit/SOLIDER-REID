# exp139 监控

## 实验信息
- 方法: `Query-Context LPCS`
- 类型: `exp135` 的 query-context 单变量升级
- 计划运行位置: 远程
- 当前状态: 二次全面审查中
- 直接对照:
  - `exp135 Corrected LPCS`
  - `exp138 Rank-Decayed LPCS`

## 启动记录

### [2026-03-21 13:55] 设计建档
- 启动原因:
  1. `exp137` 说明更激进的 hard ranking 会伤害 `R1`
  2. 因而除了改损失聚合，还应并行验证另一个不同创新点：
     - 当前 scorer 是否缺少 query-level context
  3. `exp139` 的目标是回答：
     - `R1` 的问题是不是来自 correction 缺少“这个 query 整体有多难”的语境
- 当前判断: 待审查
- 原因:
  - 按用户规则，训练前必须先完成 Claude 全面审查

### [2026-03-21 14:02] Claude 全面审查已启动
- 审查文件:
  - `experiments/exp139/claude_review.md`
- 审查请求:
  - `experiments/exp139/claude_review_request.txt`
- 审查范围:
  1. `design.md`
  2. config
  3. `defaults.py`
  4. `processor.py`
  5. `pose_backbone_model.py`
  6. `pair_adaptive_fusion.py`
  7. 与 `exp135` 的单变量对照关系
- 运行方式:
  - 使用 PTY 会话启动，避免 `nohup` 模式下 `claude` 进程空退出
- 当前判断: 审查进行中，暂不启动训练
- 原因:
  - 用户明确要求先完成全面审查，再由用户告知审查结束

### [2026-03-21 14:24] Claude 全面审查未通过，禁止启动
- 审查结论:
  - `experiments/exp139/claude_review.md` 明确给出“不允许启动”
- Blocking:
  1. 测试时 `cvk_residual` 路径仍只构造 6 维 descriptor，而 `query_ctx` 版本 head 期望 11 维输入，`epoch 10` eval 必崩
  2. 当前 context 使用 `row_pos_mean / row_neg_mean / row_margin` 等 label-dependent 统计，测试阶段无法构造，属于设计缺陷，不是单纯实现漏接
- 当前判断: 停止启动，先重构
- 原因:
  - 必须先把 query context 改成 train/test 一致、且不依赖 label 的版本，再重新送 Claude 全面审查

### [2026-03-21 14:28] 按审查意见重构为无标签 query context
- 重构目标:
  1. 去掉 `row_pos_mean / row_neg_mean / row_margin` 这类 label-dependent 统计
  2. 改成训练和测试都能直接构造的 query-level context
  3. 让 evaluator 与训练共用同一套 11 维 descriptor 语义
- 新版 context 5 维:
  - `row_mean`
  - `row_std`
  - `row_min`
  - `row_support_mean`
  - `row_gap_mean`
- 当前判断: 重构完成，准备二次全面审查
- 原因:
  - 这次修的不是小接线，而是把 `exp139` 从“oracle context”改成真正 retrieval-time 可用的 context

### [2026-03-21 14:31] 二次自检通过，准备重新送 Claude 全面审查
- 自检结果:
  1. `processor.py / utils/metrics.py / pair_adaptive_fusion.py / pose_backbone_model.py` 均通过 `py_compile`
  2. 11 维 descriptor 最小样例检查通过：
     - `base desc = [3, 5, 6]`
     - `query ctx = [3, 5, 5]`
     - `concat desc = [3, 5, 11]`
  3. config 仍保持单变量：
     - `POSE_LPCS_CONTEXT_MODE='query_ctx'`
     - `POSE_TEST_FEAT='cvk_residual'`
- 下一步:
  - 使用 `claude_review_request_v2.txt` 发起第二轮全面审查
- 当前判断: 待二审
- 原因:
  - 只有二审明确放行后，才允许把这条线发到远程服务器

### [2026-03-21 14:33] Claude 二次全面审查已启动
- 审查文件:
  - `experiments/exp139/claude_review_v2.md`
- 审查请求:
  - `experiments/exp139/claude_review_request_v2.txt`
- 重点核查:
  1. 当前 query context 是否已完全去 label 依赖
  2. `processor.py` 与 `utils/metrics.py` 是否共用同一 11 维 descriptor 语义
  3. train/test `cvk_residual` 路径是否已经闭环
- 当前判断: 二审进行中，远程继续等待
- 原因:
  - 用户要求全面审查完成后再放行远程训练
