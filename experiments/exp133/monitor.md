# exp133 监控

## 实验信息
- 方法: `LPCS`（Learned Pair Correction Scorer）
- 类型: `exp132` 之后的新主线候选
- 运行位置: 未启动
- 当前状态: 仅完成设计建档，等待实现与 Claude 审查
- 直接对照:
  - `exp132a cvk_adaptive`
  - `exp132b cvk_hybrid`
  - `exp125`

## 启动记录

### [2026-03-21 07:25] 设计建档
- 启动原因:
  1. `exp132` 已较干净地否定第一版 `alpha-fusion`
  2. learned pair module 的大方向仍未被否定
  3. 当前最合理的升级是：
     - 从“学该信谁”
     - 转到“学该修正多少”
- 当前判断: 待实现
- 原因:
  - 先完成设计收束，再按用户规则做 Claude 审查，审查通过后才允许启动
