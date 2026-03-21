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

### [2026-03-22 00:42] `exp140` 已进入真实迭代，warmup 前段启动健康
- 当前进度:
  - 已完成 `Epoch 1`
  - 当前进入 `Epoch 2`
- 关键训练日志:
  - `Epoch[1] Iter[20/227] Loss: 22.003`
  - `Epoch[1] Iter[200/227] Loss: 15.641`
  - `Epoch 1 done. Time per epoch: 60.275[s]`
  - `Epoch[2] Iter[60/227] Loss: 12.264`
- 形态观察:
  1. warmup 前段 loss 正常下降，没有 NaN / Inf
  2. 训练速度约 `223 samples/s`，与近期本地主线相符
  3. 当前仍处于 `LPCS warmup=20` 内，尚未进入 `residual_conf` 真正激活区
- 当前判断: 继续
- 原因:
  - 现在只能下“启动健康”的结论；真正有信息量的仍是 `ep10/20` 与 `epoch 21+` 后新增的 `lpcs_cf / lpcs_ctm / lpcs_cl`

### [2026-03-22 01:03] `exp140` 到 `ep20`：warmup 形态健康，但机制阶段尚未真正开始
- 新验证点:
  - `ep10 = 37.1 / 50.7`
  - `ep20 = 46.8 / 59.0`
- 对照:
  - `exp135 ep10/20 = 36.7 / 50.5, 46.7 / 58.7`
  - `exp139 ep20 = 47.6 / 60.0`
- 形态观察:
  1. `ep10/20` 相对 `exp135` 基本同形，说明 warmup 没有拖坏主训练
  2. 当前仍未进入 `residual_conf` 真正发挥作用的区间
- 当前判断: 继续，等待 `epoch 21+`
- 原因:
  - `confidence calibration` 的价值只能在后 warmup 阶段通过 `lpcs_cf / lpcs_ctm / lpcs_cl` 来判断

### [2026-03-22 01:22] `exp140` 首轮 run 判为实现失效，不解读为机制负结果
- 失效节点:
  - `Epoch 20` 验证后
- 已知有效结果:
  - `ep10 = 37.1 / 50.7`
  - `ep20 = 46.8 / 59.0`
- 失效原因:
  1. `epoch 21+` 一进入 `confidence loss` 计算即崩溃
  2. 根因不是方法逻辑，而是实现使用了：
     - `sigmoid(conf_head)`
     - `F.binary_cross_entropy(...)`
     在 AMP/autocast 下不安全
  3. 运行时报错已确认：
     - `binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast`
- 当前处理:
  1. 立即停止把这轮 run 当作有效实验结果
  2. 将 `PairResidualConfidenceScorer` 改成输出 `conf_logits`
  3. 训练端改用 `binary_cross_entropy_with_logits`
  4. 测试端继续使用 `sigmoid(conf_logits) * raw_delta`
- 当前判断: 本轮无效，准备 clean rerun
- 原因:
  - 这次失败是后 warmup 才暴露的实现问题，不能被解释成 `confidence-calibrated correction` 本身无效

### [2026-03-22 01:25] `exp140` 修复已完成，并已重新发起全面 Claude 审查
- 修复内容:
  1. `PairResidualConfidenceScorer` 改为输出 `conf_logits`
  2. 训练端 `confidence loss` 改用 `binary_cross_entropy_with_logits`
  3. 测试端保持：
     - `delta = sigmoid(conf_logits) * raw_delta`
- 本地自检:
  1. `py_compile` 已通过：
     - `model/modules/pair_adaptive_fusion.py`
     - `processor/processor.py`
     - `utils/metrics.py`
  2. 最小样例检查通过：
     - `delta_shape = (5,)`
     - `conf_logits_shape = (5,)`
     - `sigmoid(conf_logits).mean() = 0.5`
- 审查文件:
  - 请求: `experiments/exp140/claude_review_request_v2.txt`
  - 输出: `experiments/exp140/claude_review_v2.md`
- 审查会话:
  - PTY session: `97863`
- 当前判断: 等待全面审查结束，不启动 clean rerun
- 原因:
  - 用户要求由用户确认审查结束后再继续启动实验

### [2026-03-22 01:31] `exp140` 二审通过，clean rerun 已重新启动
- 审查文件:
  - `experiments/exp140/claude_review_v2.md`
- 审查结论:
  - **允许启动**
- 启动说明:
  1. 按原计划先尝试后台 `nohup` 启动到：
     - `log/occluded_duke/exp140_lpcs_conf_rerun1`
     但该壳层启动没有留下有效主进程，且 `nohup.log` 为空
  2. 随后改用前台探针确认修复版是否真的可跑
  3. 探针已稳定进入真实训练，因此直接将其提升为本次官方 clean rerun：
     - `OUTPUT_DIR=./log/occluded_duke/exp140_lpcs_conf_rerun1_probe`
     - 会话 ID: `92668`
- 启动确认:
  1. 配置日志已确认：
     - `POSE_LPCS_HEAD_MODE: residual_conf`
     - `POSE_TEST_FEAT: cvk_residual`
  2. 模型日志已确认：
     - `[LPCS] Learned Pair Correction Scorer enabled: head_mode=residual_conf ...`
  3. 训练日志已确认：
     - `[LPCS] enabled: ... head_mode=residual_conf, conf_weight=0.25 ...`
  4. 已进入真实迭代：
     - `Epoch[1] Iter[20/227] Loss: 22.003`
     - `Epoch[1] Iter[40/227] Loss: 20.455`
- 当前判断: 继续，重新进入 warmup 观察期
- 原因:
  - 二审已明确放行
  - 修复后的 logits 版本已经证明可以正常进入训练，不再存在 `epoch 21+` 前的启动阻塞

### [2026-03-22 02:00] `exp140` clean rerun 已越过 warmup，confidence calibration 首次真实生效
- 新验证点:
  - `ep10 = 37.1 / 50.7`
  - `ep20 = 46.8 / 59.5`
- 对照:
  - `exp135 ep10/20 = 36.7 / 50.5, 46.7 / 58.7`
  - `exp139 ep20 = 47.6 / 60.0`
- 后 warmup 机制信号:
  1. `lpcs_cf` 从约 `0.588` 持续升到 `0.730`
  2. `lpcs_ctm` 维持在约 `0.098 ~ 0.103`
  3. `lpcs_cl` 从约 `0.530` 降到 `0.328`
  4. `lpcs_dm / lpcs_rdm` 从约 `0.009 / 0.016` 升到 `0.053 / 0.072`
  5. `lpcs_ctxm = 0.000`，符合本实验 `context_mode=none` 的设计
- 形态观察:
  1. 这次 `confidence calibration` 已经确认真实接入，不再是失效 run
  2. `ep20` 相对 `exp135` 是弱正向
  3. 但当前 `conf_mean` 明显高于 `conf_target_mean`，说明这版 gate 暂时偏激进而非保守
- 当前判断: 继续，优先看 `ep30`
- 原因:
  - 现在终于第一次拿到了有效的 `confidence calibration` 机制证据
  - 但还需要 `ep30` 来判断这种高 `conf` 形态究竟会兑现成收益，还是只是把 correction 放大得过早
