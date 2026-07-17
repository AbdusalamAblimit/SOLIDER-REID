# exp128 监控

## 实验信息
- 方法: Exact Top-K Pair SCRD + Freeze30
- 类型: 训练端组合验证（基于两条已获正信号的机制）
- 运行位置: 本地 3090
- 主配置: `configs/occluded_duke/pose_psg_gcn_pair_top_exact_scrd_freeze30.yml`
- 核心变量: 相对 `exp126` 只新增 `POSE_CSRD_ST_UPDATE_STOP_EPOCH = 30`
- 直接对照: `exp126`
- 次对照: `exp125`, `exp121`

## 启动记录

### [2026-03-20 18:11] 启动前记录
- 启动原因:
  1. `exp127` 到 `ep100` 已证明 feature-level residual completion 不是当前主突破口
  2. 现在本地应回到当前最有希望的 relational 主线
  3. `freeze30` 是已验证的 supporting mechanism，因此值得接到 exact-topk 稀疏版本上做 full-model 候选验证
- 当前判断: 待启动

### [2026-03-20 18:31] 手动停表
- 停止原因:
  1. 用户明确要求不要继续 `freeze` 线
  2. `freeze20 / freeze30` 的既有证据已经足够说明它只是弱 supporting mechanism，不值得继续消耗本地 3090
- 当前判断: 终止
- 原因:
  - 本地算力改投非 `freeze` 主线；`exp128` 不进入正式比较
