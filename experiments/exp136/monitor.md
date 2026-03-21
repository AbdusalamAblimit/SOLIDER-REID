# exp136 监控

## 实验信息
- 方法: `Corrected Changed-Pair Sparse LPCS`
- 类型: `exp134` 失效 run 后的共享接线修复重跑
- 运行位置: 待启动（远程）
- 当前状态: 已完成设计建档，待 Claude 审查
- 直接对照:
  - `exp135 Corrected LPCS`

## 启动记录

### [2026-03-21 08:58] 设计建档
- 启动原因:
  1. `exp134` 被确认是失效 run，当前还没有真正测到 sparse `LPCS`
  2. 在修复共享接线 bug 后，远程应并行验证 corrected sparse `LPCS`
  3. 这样本地/远程就能形成 clean 的 `full vs sparse` 直接对照
- 当前判断: 待审查
- 原因:
  - 按用户规则，启动前必须再次通过 Claude 审查

### [2026-03-21 09:01] Claude 审查通过
- 审查文件:
  - `experiments/exp136/claude_review.md`
- 审查结论:
  - **允许启动**
- 审查确认:
  1. 共享接线 bug 已真正修复
  2. 相对 `exp135` 仅新增：
     - `POSE_LPCS_PAIR_MODE='delta_top'`
     - `POSE_LPCS_PAIR_TOP_RATIO=0.25`
  3. `lpcs_psr / lpcs_pf` 足以验证 sparse routing 是否真实生效
- 当前判断: 允许启动
- 原因:
  - 已满足远程 clean rerun 条件，可作为 `exp135` 的直接并行对照
