# exp139 监控

## 实验信息
- 方法: `Query-Context LPCS`
- 类型: `exp135` 的 query-context 单变量升级
- 计划运行位置: 远程
- 当前状态: 已完成设计与代码接线，等待 Claude 全面审查
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
