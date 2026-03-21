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

### [2026-03-22 02:55] 首轮 Claude 审查未通过，已确认是 config 模板错误
- 审查文件:
  - `experiments/exp141/claude_review.md`
- blocking 结论:
  1. 当前 `pose_psg_gcn_lpcs_comp_ctx.yml` 不是从 `exp135` config 严格继承
  2. 至少存在多处关键字段差异，会破坏单变量原则
  3. 其中包含 `MODEL.NAME` 等高风险项，不能放行
- 当前处理:
  1. 按审查建议，将 `exp141` config 改成严格复制 `exp135`
  2. 仅保留两处差异：
     - `POSE_LPCS_CONTEXT_MODE: 'comp_ctx'`
     - `OUTPUT_DIR`
- 当前判断: 修复完成，准备二次全面审查
- 原因:
  - 审查否掉的不是 `comp_ctx` 机制本身，而是实验隔离性不干净

### [2026-03-22 02:57] 二次自检通过，已满足重新送审条件
- 自检结果:
  1. 当前 config 相对 `exp135` 的 diff 只剩两处：
     - `POSE_LPCS_CONTEXT_MODE: 'comp_ctx'`
     - `OUTPUT_DIR`
  2. 这意味着 `exp141` 现在已满足单变量前提
- 审查文件:
  - 请求: `experiments/exp141/claude_review_request_v2.txt`
  - 输出: `experiments/exp141/claude_review_v2.md`
- 当前判断: 等待二次全面审查，不启动训练
- 原因:
  - 用户要求由用户确认审查结束后再继续
