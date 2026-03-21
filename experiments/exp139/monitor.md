# exp139 监控

## 实验信息
- 方法: `Query-Context LPCS`
- 类型: `exp135` 的 query-context 单变量升级
- 计划运行位置: 远程
- 当前状态: 远程训练中
- 直接对照:
  - `exp135 Corrected LPCS`
  - `exp138 Rank-Decayed LPCS`

## 启动记录

### [2026-03-21 13:55] 设计建档
- 启动原因:
  1. `exp137` 说明更激进的 hard ranking 会伤害 `R1`
  2. 因而除了改损失聚合，还应并行验证另一个不同创新点：
     - 当前 scorer 是否缺少 query-level context
  3. `exp139` 的目标是回答：
     - `R1` 的问题是不是来自 correction 缺少“这个 query 整体有多难”的语境
- 当前判断: 待审查
- 原因:
  - 按用户规则，训练前必须先完成 Claude 全面审查

### [2026-03-21 14:02] Claude 全面审查已启动
- 审查文件:
  - `experiments/exp139/claude_review.md`
- 审查请求:
  - `experiments/exp139/claude_review_request.txt`
- 审查范围:
  1. `design.md`
  2. config
  3. `defaults.py`
  4. `processor.py`
  5. `pose_backbone_model.py`
  6. `pair_adaptive_fusion.py`
  7. 与 `exp135` 的单变量对照关系
- 运行方式:
  - 使用 PTY 会话启动，避免 `nohup` 模式下 `claude` 进程空退出
- 当前判断: 审查进行中，暂不启动训练
- 原因:
  - 用户明确要求先完成全面审查，再由用户告知审查结束

### [2026-03-21 14:24] Claude 全面审查未通过，禁止启动
- 审查结论:
  - `experiments/exp139/claude_review.md` 明确给出“不允许启动”
- Blocking:
  1. 测试时 `cvk_residual` 路径仍只构造 6 维 descriptor，而 `query_ctx` 版本 head 期望 11 维输入，`epoch 10` eval 必崩
  2. 当前 context 使用 `row_pos_mean / row_neg_mean / row_margin` 等 label-dependent 统计，测试阶段无法构造，属于设计缺陷，不是单纯实现漏接
- 当前判断: 停止启动，先重构
- 原因:
  - 必须先把 query context 改成 train/test 一致、且不依赖 label 的版本，再重新送 Claude 全面审查

### [2026-03-21 14:28] 按审查意见重构为无标签 query context
- 重构目标:
  1. 去掉 `row_pos_mean / row_neg_mean / row_margin` 这类 label-dependent 统计
  2. 改成训练和测试都能直接构造的 query-level context
  3. 让 evaluator 与训练共用同一套 11 维 descriptor 语义
- 新版 context 5 维:
  - `row_mean`
  - `row_std`
  - `row_min`
  - `row_support_mean`
  - `row_gap_mean`
- 当前判断: 重构完成，准备二次全面审查
- 原因:
  - 这次修的不是小接线，而是把 `exp139` 从“oracle context”改成真正 retrieval-time 可用的 context

### [2026-03-21 14:31] 二次自检通过，准备重新送 Claude 全面审查
- 自检结果:
  1. `processor.py / utils/metrics.py / pair_adaptive_fusion.py / pose_backbone_model.py` 均通过 `py_compile`
  2. 11 维 descriptor 最小样例检查通过：
     - `base desc = [3, 5, 6]`
     - `query ctx = [3, 5, 5]`
     - `concat desc = [3, 5, 11]`
  3. config 仍保持单变量：
     - `POSE_LPCS_CONTEXT_MODE='query_ctx'`
     - `POSE_TEST_FEAT='cvk_residual'`
- 下一步:
  - 使用 `claude_review_request_v2.txt` 发起第二轮全面审查
- 当前判断: 待二审
- 原因:
  - 只有二审明确放行后，才允许把这条线发到远程服务器

### [2026-03-21 14:33] Claude 二次全面审查已启动
- 审查文件:
  - `experiments/exp139/claude_review_v2.md`
- 审查请求:
  - `experiments/exp139/claude_review_request_v2.txt`
- 重点核查:
  1. 当前 query context 是否已完全去 label 依赖
  2. `processor.py` 与 `utils/metrics.py` 是否共用同一 11 维 descriptor 语义
  3. train/test `cvk_residual` 路径是否已经闭环
- 当前判断: 二审进行中，远程继续等待
- 原因:
  - 用户要求全面审查完成后再放行远程训练

### [2026-03-21 14:34] Claude 二次全面审查通过，允许启动
- 审查文件:
  - `experiments/exp139/claude_review_v2.md`
- 审查结论:
  - **允许启动**
- 审查确认:
  1. 原始两个 blocking 已全部修复
  2. 当前 `query_ctx` 已改成无标签、train/test 对称的 11 维 descriptor
  3. `processor.py` 与 `utils/metrics.py` 已共用同一套 context 语义
- 非阻塞提醒:
  1. 训练里另有一个用于 pair weighting 的 `pair_change=|teacher_dist-base_dist|`，与 context 用的 `row_gap_mean` 语义不同，后续解释时要避免混淆
  2. 训练排除对角线、测试不过滤对角线是合理设计，但目前只在代码逻辑里体现，没额外注释
- 当前判断: 放行，准备远程启动
- 原因:
  - 这条线现在终于可以作为真正的第二创新点被远程干净验证

### [2026-03-21 14:36] 远程 `exp139` 正式启动
- 远程机器:
  - 恒源云 `5060 Ti`
- 同步动作:
  1. 本地已 push 到 `origin/exp/pose_heatmap`
  2. 远程已 `git pull origin exp/pose_heatmap`
- 启动命令:
  - `python3 train.py --config_file configs/occluded_duke/pose_psg_gcn_lpcs_query_ctx.yml OUTPUT_DIR ./log/occluded_duke/exp139_lpcs_query_ctx`
- 远程输出:
  - `log/occluded_duke/exp139_lpcs_query_ctx/remote_nohup.log`
- 启动确认:
  1. 配置日志已确认：
     - `POSE_LPCS_CONTEXT_MODE: query_ctx`
     - `POSE_TEST_FEAT: cvk_residual`
  2. 远程训练主进程已存在：
     - `python3 train.py --config_file configs/occluded_duke/pose_psg_gcn_lpcs_query_ctx.yml ...`
  3. 远程 GPU 已被占用：
     - `5060 Ti` 显存约 `6692 MiB`
- 当前判断: 继续
- 原因:
  - 这轮是当前唯一真正测试“无标签 query-level context 是否改善 pair correction”的 clean run

### [2026-03-21 14:48] 远程 warmup 前段运行健康，但尚未进入有效判别区
- 当前进度:
  - 已完成 `Epoch 1-8`
  - 当前处于 `Epoch 9`
- 关键训练日志:
  - `Epoch 1 done. Time per epoch: 92.830[s]`
  - `Epoch 5 done. Time per epoch: 91.689[s]`
  - `Epoch 8 done. Time per epoch: 93.728[s]`
- 形态观察:
  1. `context_mode=query_ctx` 已稳定打印，无接线缺失
  2. 早期 loss 下降形状与 `exp135` 同向，没有异常爆炸
  3. 远程单 epoch 约 `92~94s`，明显慢于本地 `exp138` 的约 `58~60s`
- 当前判断: 继续
- 原因:
  - 当前还未到 `ep10`，更未越过 `LPCS warmup=20`
  - 现在只能下“启动健康”的结论，真正关键的是 `ep10/20` 与 `epoch 21+` 后 `lpcs_ctxm` 是否显著大于 `0`

### [2026-03-21 22:52] `exp139` 首个验证点：启动健康，但早期略弱于 `exp135`
- 新验证点:
  - `ep10 = 36.5 / 50.0`
- 对照:
  - `exp135 ep10 = 36.7 / 50.5`
  - `exp138 ep10 = 36.7 / 50.5`
- 运行形态:
  1. 远程每个 epoch 约 `92~94s`
  2. 当前已完成 `Epoch 13`
  3. 日志持续确认 `context_mode=query_ctx`
- 当前判断: 继续，暂不做负面解读
- 原因:
  - 这仍处于 `LPCS warmup=20` 内，`query_ctx` 的真实效果尚未进入验证期
  - 真正的关键节点是 `ep20`，以及 `epoch 21+` 后 `lpcs_ctxm` 是否显著大于 `0`

### [2026-03-21 23:27] `exp139` 越过 warmup：query-context 已真实接入，且早中期为正信号
- 新验证点:
  - `ep20 = 47.6 / 60.0`
- 对照:
  - `exp135 ep20 = 46.7 / 58.7`
  - `exp138 ep20 = 46.7 / 58.7`
  - `exp030a ep20 = 46.8 / 60.9`
- 关键机制信号:
  1. `epoch 21+` 后 `lpcs_ctxm` 稳定在 `0.408 ~ 0.430`
  2. `lpcs_fg > lpcs_bg`，到 `epoch 28` 约为 `0.485 > 0.392`
  3. `lpcs_dm / lpcs_ds` 已抬到约 `0.112 / 0.029`
  4. `lpcs_rwm = 1.000`，符合本实验 `rank_mode=all` 的设计预期
- 运行代价:
  - `epoch 21+` 后单轮时间从约 `93s` 升到 `132~137s`
- 当前判断: 继续，并上调为当前双线中的优先观察对象
- 原因:
  - 这条线第一次把 query-level context 干净接入了 `LPCS`
  - 而且到 `ep20` 为止已经相对 `exp135/138` 给出清晰正信号

### [2026-03-21 23:27] `exp139` 激活后机制继续稳定，当前仍是主候选
- 当前进度:
  - 已到 `Epoch 28`
- 机制延续观察:
  1. `lpcs_ctxm` 持续稳定在 `0.408 ~ 0.430`
  2. `lpcs_fg > lpcs_bg`，到 `epoch 28` 约为 `0.485 > 0.392`
  3. `lpcs_dm / lpcs_ds` 已升到约 `0.112 / 0.029`
  4. `lpcs_rwm = 1.000`，继续符合 `rank_mode=all`
- 当前判断: 继续，保持主候选优先级
- 原因:
  - 目前没有任何信号表明 `query_ctx` 只是偶然抖动
  - 下一次真正决定性的节点是 `ep30`

### [2026-03-22 00:14] `exp139` 到 `ep40`：query-context 持续转正，当前升为唯一主候选
- 新验证点:
  - `ep40 = 57.0 / 68.8`
- 对照:
  - `exp135 ep40 = 56.7 / 68.3`
  - `exp138 ep40 = 56.8 / 68.6`
  - `exp030a ep40 = 55.6 / 68.6`
- 关键机制信号:
  1. `lpcs_ctxm` 已从 `0.408 ~ 0.430` 继续抬到约 `0.459 ~ 0.467`
  2. `lpcs_fg > lpcs_bg` 的差距持续扩大，到 `epoch 46` 左右约为 `1.36 > 0.53`
  3. `lpcs_dm / lpcs_ds` 已升到约 `0.36 / 0.21`
  4. `lpcs_rwm = 1.000`，继续符合 `rank_mode=all` 的设计预期
- 运行代价:
  - `epoch 37~46` 单轮大约 `138~144s`
- 当前判断: 继续，当前为两条升级线中唯一的主候选
- 原因:
  - 到 `ep40` 为止，`query_ctx` 已同时超过 `exp135` 与 `exp138`
  - 而且这次优势与 `lpcs_ctxm` 持续上升、`lpcs_fg > lpcs_bg` 扩大是同步出现的，不像偶然抖动

### [2026-03-22 00:39] `exp139` 到 `ep50`：继续稳定领先，query-context 主候选地位强化
- 新验证点:
  - `ep50 = 58.7 / 70.4`
- 对照:
  - `exp135 ep50 = 57.8 / 69.5`
  - `exp138 ep50 = 57.9 / 69.5`
  - `exp030a ep50 = 55.7 / 68.8`
- 关键机制信号:
  1. `lpcs_ctxm` 已稳定抬到约 `0.465 ~ 0.473`
  2. `lpcs_fg > lpcs_bg` 继续扩大，到 `epoch 56` 左右约为 `1.49 > 0.57`
  3. `lpcs_dm / lpcs_ds` 已升到约 `0.41 / 0.22`
  4. `lpcs_rwm = 1.000`，继续符合 `rank_mode=all`
- 当前判断: 继续，当前仍为全局主候选
- 原因:
  - 到 `ep50` 为止，`query_ctx` 不只是早期正信号，而是已经相对 `exp135/138` 拉开接近 `+0.8~0.9 mAP / +0.9 R1`
  - 这使它首次开始具备“可能成为论文主机制”的形态

### [2026-03-22 01:20] `exp139` 到 `ep70`：中后期仍维持主候选，但当前增益更偏向 mAP
- 新验证点:
  - `ep60 = 57.9 / 69.0`
  - `ep70 = 59.5 / 71.0`
- 对照:
  - `exp135 ep60 = 58.4 / 69.4`
  - `exp135 ep70 = 59.0 / 70.9`
  - `exp030a ep70 = 58.1 / 70.9`
- 关键机制信号:
  1. `lpcs_ctxm` 继续稳定在 `0.474 ~ 0.480`
  2. `lpcs_fg > lpcs_bg` 维持明显差距，约 `1.54 ~ 1.57 > 0.59 ~ 0.61`
  3. `lpcs_dm / lpcs_ds` 已稳定在约 `0.42 ~ 0.43 / 0.215 ~ 0.220`
  4. `lpcs_wm` 大多在 `0.97 ~ 1.00`
- 当前判断: 继续，仍是当前唯一主候选
- 原因:
  - 到 `ep70` 为止，`query_ctx` 依然稳住了相对 `exp135` 的优势
  - 但当前形态更像 `mAP` 稳定更强、`R1` 小幅领先，后续仍需看 `ep80/90` 是否能把优势进一步坐实
