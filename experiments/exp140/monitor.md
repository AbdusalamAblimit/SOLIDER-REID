# 实验 exp140: Confidence-Calibrated LPCS

## 设计链接

- [design.md](./design.md)

## 监控记录

### [2026-03-22 00:20] 本地候选 `exp140` 已实现，等待全面 Claude 审查
- 当前状态:
  - 代码已接线完成，尚未启动训练
- 核心改动:
  1. `PairResidualScorer` 扩展出 `residual_conf` 版本
  2. `LPCS` 训练路径支持：
     - `raw_delta`
     - `conf`
     - `conf_target`
     - `conf_loss`
  3. `cvk_residual` 测试路径已支持 `delta = conf * raw_delta`
- 当前判断: 待审查
- 原因:
  - 用户要求所有新实验必须先通过 Claude 审查
  - 这轮要重点审查 train/test 对称性、loss 是否真正单变量、以及 confidence target 是否引入隐藏变量

### [2026-03-22 00:23] 本地自检通过，Claude 全面审查已启动
- 自检结果:
  1. `python -m py_compile` 已通过：
     - `model/modules/pair_adaptive_fusion.py`
     - `model/pose_backbone_model.py`
     - `processor/processor.py`
     - `utils/metrics.py`
  2. 最小前向已通过：
     - `PairResidualConfidenceScorer` 可正常输出 `(delta, conf)`
     - 初始 `conf` 为 `0.5`，符合零初始化预期
  3. config 合并已通过：
     - `POSE_LPCS=True`
     - `POSE_LPCS_HEAD_MODE='residual_conf'`
     - `OUTPUT_DIR=./log/occluded_duke/exp140_lpcs_conf`
- 审查进程:
  - `claude` 后台 PID: `1220046`
  - 输出文件: `experiments/exp140/claude_review.md`
- 当前判断: 等待审查完成，不启动训练
- 原因:
  - 用户明确要求“所有新实验通过 Claude 审查后才能开始”
