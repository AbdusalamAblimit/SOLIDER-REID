# exp129 监控

## 实验信息
- 方法: Residual-Correction SCRD
- 类型: 训练端单变量改进
- 运行位置: 本地 3090
- 主配置: `configs/occluded_duke/pose_psg_gcn_pair_top_resid_scrd.yml`
- 核心变量: 相对 `exp125` 只新增 `POSE_CSRD_TARGET_MODE = 'residual'`
- 直接对照: `exp125`
- 次对照: `exp123`, `exp120`

## 启动记录

### [2026-03-20 18:35] 启动前记录
- 启动原因:
  1. 用户已明确否定继续试 `freeze`；`exp128` 已手动停止
  2. `exp127` 已说明 feature-level direct completion 不是当前主突破口
  3. `exp120/123/125` 的共同缺口是：support-complete teacher 的新增 correction 很可能被完整 teacher target 稀释
- 当前判断: 待启动
- 原因:
  - `exp129` 是相对 `exp125` 的下一跳：保留当前最强的在线 relational 主线，只改 distillation target，从 full teacher 改成 residual correction
