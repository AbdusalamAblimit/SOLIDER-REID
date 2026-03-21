# exp138 监控

## 实验信息
- 方法: `Rank-Decayed LPCS`
- 类型: `exp135` 的平滑 top-sensitive 单变量升级
- 计划运行位置: 本地
- 当前状态: 已完成设计与代码接线，等待 Claude 全面审查
- 直接对照:
  - `exp135 Corrected LPCS`
  - `exp137 Hard-Rank LPCS`

## 启动记录

### [2026-03-21 13:55] 设计建档
- 启动原因:
  1. `exp136` 已证明真稀疏 routing 不是主突破口
  2. `exp137` 已证明 hard-top 25% rank selection 太激进
  3. 因而当前更合理的下一步是：
     - 保留 full-pair 上下文
     - 但用更平滑的方式强调 top-ranked mistakes
- 当前判断: 待审查
- 原因:
  - 按用户规则，训练前必须先完成 Claude 全面审查

### [2026-03-21 14:02] Claude 全面审查已启动
- 审查文件:
  - `experiments/exp138/claude_review.md`
- 审查请求:
  - `experiments/exp138/claude_review_request.txt`
- 审查范围:
  1. `design.md`
  2. config
  3. `defaults.py`
  4. `processor.py`
  5. `pose_backbone_model.py`
  6. `pair_adaptive_fusion.py`
  7. 与 `exp135 / exp137` 的单变量对照关系
- 运行方式:
  - 使用 PTY 会话启动，避免 `nohup` 模式下 `claude` 进程空退出
- 当前判断: 审查进行中，暂不启动训练
- 原因:
  - 用户明确要求先完成全面审查，再由用户告知审查结束
