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

### [2026-03-22 00:24] Claude 审查已改为 PTY 长会话重启
- 纠正说明:
  1. 首次后台 `nohup` 审查命令没有真正留下有效进程
  2. 已改用 PTY 会话重启，避免引号拼接导致的静默失败
- 当前状态:
  - 审查会话仍在运行
  - 会话 ID: `34526`
  - 输出文件: `experiments/exp140/claude_review.md`
- 当前判断: 继续等待审查结果
- 原因:
  - 现在这条审查链路已经稳定，后续只需等待用户告知“review 已结束”

### [2026-03-22 00:39] Claude 审查通过，`exp140` 已正式启动
- 审查文件:
  - `experiments/exp140/claude_review.md`
- 审查结论:
  - **允许启动**
- 审查关键结论:
  1. 相对 `exp135` 是真正单变量
  2. `residual_conf` 在 train/test 两侧都完整生效
  3. `conf_target` 不引入 label/oracle 泄漏
- 启动说明:
  1. 前两次后台启动均失败，但属于壳层问题，不是代码问题：
     - 一次是未先创建 `OUTPUT_DIR`
     - 一次是后台 shell 走错 Python 环境
  2. 最终已改用绝对解释器路径在 PTY 会话中正式启动：
     - `/root/miniconda3/envs/solider-reid/bin/python train.py --config_file configs/occluded_duke/pose_psg_gcn_lpcs_conf.yml ...`
  3. 当前训练会话已进入真实启动阶段：
     - 会话 ID: `68362`
     - 输出目录: `log/occluded_duke/exp140_lpcs_conf`
- 启动确认:
  1. 配置日志已确认 `POSE_LPCS_HEAD_MODE: residual_conf`
  2. 模型日志已确认：
     - `[LPCS] Learned Pair Correction Scorer enabled: head_mode=residual_conf ...`
  3. 训练日志已确认：
     - `[LPCS] enabled: ... head_mode=residual_conf, conf_weight=0.25 ...`
- 当前判断: 继续，进入 warmup 观察期
- 原因:
  - 这轮终于开始第一次真实测试“pair correction 是否需要 confidence calibration”
