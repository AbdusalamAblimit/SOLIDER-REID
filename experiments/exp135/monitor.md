# exp135 监控

## 实验信息
- 方法: `Corrected LPCS Clean Rerun`
- 类型: `exp133` 失效 run 后的共享接线修复重跑
- 运行位置: 待启动（本地）
- 当前状态: 已完成设计建档，待 Claude 审查
- 直接对照:
  - `exp132 LTCS`
  - `exp136 Corrected Sparse LPCS`

## 启动记录

### [2026-03-21 08:58] 设计建档
- 启动原因:
  1. `exp133` 被确认是失效 run，不能用于 LPCS 判断
  2. 当前最紧要的不是切题，而是第一次把真正激活的 `LPCS` 测起来
  3. 本地应优先重跑 corrected `LPCS`，作为 `exp136` 的 clean 直接对照
- 当前判断: 待审查
- 原因:
  - 按用户规则，启动前必须再次通过 Claude 审查

### [2026-03-21 09:02] Claude 审查通过
- 审查文件:
  - `experiments/exp135/claude_review.md`
- 审查结论:
  - **允许启动**
- 审查确认:
  1. 共享接线 bug 已真正修复
  2. `LPCS` loss 会在 `epoch 21+` 真正进入训练
  3. `lpcs_*` 会在日志中出现
  4. 当前相对 intended `exp133` 仅是共享 bug 修复 + 新输出目录，满足 clean rerun 的单变量原则
- 当前判断: 允许启动
- 原因:
  - 这轮终于有资格第一次真正测试 `LPCS`
