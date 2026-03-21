# 实验 exp142: SKC（Support-Conditioned Keypoint Completion）

## 2026-03-21 19:02 启动前记录

- 状态：仅完成设计，未改代码，未启动训练
- 当前定位：本地主线从 `LPCS` 小变体切换到 feature-space support completion 大改动
- 原因：
  1. `exp109` 已说明真正 headroom 在 `support incomplete`
  2. `exp119-140` 的 pair correction 系列已经提供了足够多“机制有用但突破有限”的证据
  3. 用户明确要求不要继续围绕一个小点耗时间
- 当前判断：
  1. `exp141` 虽已完成二次审查，但仍属于 `LPCS` 家族增量线，暂不启动
  2. `exp142` 将作为本地下一条真正不同的创新点
  3. 下一步先改代码，再做全面 Claude 审查

## 预设监控清单

后续真正启动训练后，每次检查除了常规 `loss / mAP / R1`，还必须补以下行为日志：

- `skc_lmr`
- `skc_spr`
- `skc_arr`
- `skc_gm`
- `skc_gs`
- `skc_dn`
- `skc_pc`
- `skc_pcnt`
- `skc_cl`
- `skc_pre`
- `skc_post`

如果启动后这些日志缺失，则本次 run 视为不可解释 run，需要优先补日志再继续。

## 2026-03-21 19:02 代码接线后自检

- 状态：已完成第一版代码接线，仍未启动训练
- 本轮变更：
  1. 新增 `POSE_SKC` 默认配置与独立 config
  2. 在 `SkeletonGCNHead` 中接入 `Support-Supervised Keypoint Completion`
  3. 在 `processor.py` 中接入：
     - SKC support bank
     - consistency loss
     - 行为日志与 support-target 日志
- 设计修正：
  1. 将最初的 “Support-Conditioned” 收紧为 “Support-Supervised”
  2. 原因是要保证 train/test 一致：
     - 模块本体只依赖当前图
     - support bank 只在训练中作为监督目标
- 自检结果：
  1. `py_compile` 已通过：
     - `model/modules/skeleton_gcn.py`
     - `model/pose_backbone_model.py`
     - `processor/processor.py`
  2. 用 `pose_psg_gcn_skc.yml` 直接构造模型已通过
     - 控制台已打印 `[SKC] Support-Supervised Keypoint Completion enabled`
  3. 最小前向检查已通过：
     - `aux_data` 中已出现
       - `skc_raw_feats`
       - `skc_completed_feats`
       - `skc_scores`
       - `skc_stats`
     - `skc_stats` 已能返回：
       - `low_ratio`
       - `applied_ratio`
       - `gate_mean`
       - `gate_std`
       - `delta_norm`
- 当前判断：
  1. 接线层面已经具备送全面 Claude 审查的条件
  2. 下一步不是启动训练，而是先做广范围审查

## 2026-03-21 19:02 全面 Claude 审查已启动

- 状态：审查中，未启动训练
- 审查方式：
  - 使用 PTY 长会话运行 `claude -p --effort max`
  - 避免后台脱壳造成“看起来启动、实际上没跑”的假状态
- 审查输入：
  - `experiments/exp142/claude_review_request.txt`
- 审查输出目标：
  - `experiments/exp142/claude_review.md`
  - `experiments/exp142/claude_review.err`
- 当前判断：
  1. `exp142` 代码与日志自检已完成
  2. 按用户规则，必须等待 Claude 审查结论后才能启动训练

## 2026-03-21 19:02 第一轮 Claude 审查完成，已按意见修复后准备二审

- 第一轮结论：
  1. 方法方向和代码主体均被放行
  2. 但指出了 1 个中优先级与 2 个低优先级问题
- 第一轮指出的问题：
  1. `applied_ratio` 分母是全部关键点，容易误判
  2. `pre_dist` 纯日志统计却建立了不必要计算图
  3. 缺少 `delta_std`，不利于检测 delta 塌缩
- 已完成修复：
  1. `skc_stats` 新增 `applied_in_low`
  2. `skc_stats` 新增 `delta_std`
  3. `processor.py` 新增对应日志：
     - `skc_ail`
     - `skc_ds`
  4. `raw_norm / pre_dist` 已移入 `torch.no_grad()`
  5. support bank 进度日志改成 warmup 前也可观察
- 修复后自检：
  1. `py_compile` 重新通过
  2. 最小前向已确认 `skc_stats` 键变为：
     - `low_ratio`
     - `applied_ratio`
     - `applied_in_low`
     - `gate_mean`
     - `gate_std`
     - `delta_norm`
     - `delta_std`
- 当前判断：
  1. 现在更适合发起第二轮定向 Claude 审查
  2. 二审通过前，仍不启动训练
