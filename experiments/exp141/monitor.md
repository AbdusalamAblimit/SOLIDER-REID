# exp141 监控

## 实验信息
- 方法: `Competition-Context LPCS`
- 类型: `exp135` 的 context 单变量升级
- 计划运行位置: 本地
- 当前状态: 等待全面 Claude 审查
- 直接对照:
  - `exp135 Corrected LPCS`
  - `exp139 Query-Context LPCS`

## 启动记录

### [2026-03-22 02:45] 设计建档并接线完成，等待全面 Claude 审查
- 启动原因:
  1. `exp139` 说明 query-level context 确实有效
  2. 但到 `ep100` 为止，它仍更像“稳住主候选”，还没有形成明显超越
  3. 当前更值得测的是：
     - 真正重要的是否不是 query 均值摘要
     - 而是 **当前 candidate 在本 query 全部候选里的相对竞争位置**
- 核心改动:
  1. 为 `LPCS` 新增 `POSE_LPCS_CONTEXT_MODE='comp_ctx'`
  2. 新增 5 维 pair-specific competition context：
     - `base_rank`
     - `kp_rank`
     - `support_rank`
     - `gain_rank`
     - `gain_zscore`
  3. 训练与测试都按 query 的 candidate set 直接构造，无标签、train/test 对称
- 当前判断: 待审查
- 原因:
  - 用户要求新实验先走全面 Claude 审查，再由用户告知审查结束后启动

### [2026-03-22 02:49] 本地自检通过，准备发起全面 Claude 审查
- 自检结果:
  1. `py_compile` 已通过：
     - `model/modules/pair_adaptive_fusion.py`
     - `model/pose_backbone_model.py`
     - `processor/processor.py`
     - `utils/metrics.py`
  2. competition context 最小样例检查通过：
     - `base_desc = [4, 4, 6]`
     - `comp_ctx = [4, 4, 5]`
     - `concat = [4, 4, 11]`
  3. `comp_ctx.abs().mean() > 0`
     - 说明不是全零接线
- 当前判断: 可以送审，但暂不启动训练
- 原因:
  - 现在已经满足“全面审查前的最小自检”要求，下一步只等 Claude 审查结论
