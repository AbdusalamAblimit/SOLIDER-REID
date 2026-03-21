# exp137 监控

## 实验信息
- 方法: `Hard-Rank LPCS`
- 类型: `exp135` 的 ranking-aligned 单变量升级
- 运行位置: 本地
- 当前状态: 训练中
- 直接对照:
  - `exp135 Corrected LPCS`
  - `exp136 Corrected Sparse LPCS`（远程进行中，仅作间接参考）

## 启动记录

### [2026-03-21 19:45] 设计建档
- 启动原因:
  1. `exp135` 已证明 corrected full-pair `LPCS` 有效，但更偏 `mAP`
  2. `exp136` 已证明真稀疏 routing 语义成立，但到 `ep70` 仍未变成更强指标
  3. 因而当前更值得优先验证的是：
     - `LPCS` 的 ranking 聚合方式是否过于平均
- 当前判断: 待代码接线与审查
- 原因:
  - 按用户硬规则，所有新实验必须先通过 Claude 审查再允许启动

### [2026-03-21 19:52] 代码接线完成并通过本地自检
- 已完成修改:
  1. `defaults.py` 新增：
     - `POSE_LPCS_RANK_MODE`
     - `POSE_LPCS_RANK_TOP_RATIO`
  2. `processor.py` 新增 hard-rank 聚合逻辑：
     - 对 routed positives 取 hardest top-ratio
     - 对 routed negatives 取 hardest top-ratio
  3. 新增日志统计：
     - `lpcs_rsr`
  4. 新配置文件：
     - `configs/occluded_duke/pose_psg_gcn_lpcs_hard_rank.yml`
- 本地自检:
  1. `python -m py_compile` 已通过
  2. 配置 merge 已确认：
     - `POSE_LPCS=True`
     - `POSE_LPCS_RANK_MODE='hard_top'`
     - `POSE_LPCS_RANK_TOP_RATIO=0.25`
- 当前判断: 等待 Claude 审查
- 原因:
  - 接线层面暂无明显错误，但按规则必须先完成外部审查再允许启动

### [2026-03-21 19:54] Claude 审查已启动
- 审查文件:
  - `experiments/exp137/claude_review.md`
- 审查重点:
  1. 相对 `exp135` 是否满足单变量原则
  2. `hard_top` 是否真的实现为 ranking aggregation，而不是又引入一层 pair routing 变量
  3. `lpcs_rsr` 是否足以证明 hard ranking 激活
  4. 默认行为是否保持不变
- 当前判断: 暂不启动训练
- 原因:
  - 不越过用户设定的“先审查、后启动”规则

### [2026-03-21 21:17] Claude 审查通过
- 审查文件:
  - `experiments/exp137/claude_review.md`
- 审查结论:
  1. **允许启动**
  2. 无 Critical / High / Medium 项
  3. 仅有两个已接受的 Low 项：
     - `_select_top` 对空张量的理论边界
     - config 中显式写出 `pair_mode='all'` 与 `top_ratio=1.0`
- 审查确认:
  1. 相对 `exp135` 满足单变量原则
  2. `hard_top` 逻辑与设计一致：
     - hardest positive = 最大距离 top-k
     - hardest negative = 最小距离 top-k
  3. 默认行为安全，旧实验在 `rank_mode='all'` 下完全不变
  4. `lpcs_rsr` 足以证明 hard ranking 是否真正激活
- 当前判断: 允许启动
- 原因:
  - 当前已满足用户设定的“先审查，后启动”规则，可进入正式后台训练

### [2026-03-21 21:59] 本地 exp137 已正式启动并确认进入训练
- 启动方式:
  - `/root/miniconda3/envs/solider-reid/bin/python -u train.py --config_file configs/occluded_duke/pose_psg_gcn_lpcs_hard_rank.yml`
- 输出目录:
  - `log/occluded_duke/exp137_lpcs_hard_rank`
- 启动确认:
  1. `train_log.txt` 已确认：
     - `start training`
     - `[LPCS] enabled: ... pair_mode=all, top_ratio=1.0, rank_mode=hard_top, rank_top_ratio=0.25 ...`
  2. 已真实进入 iteration：
     - `Epoch[1] Iter[20/227]`
     - `Epoch[1] Iter[40/227]`
     - `Epoch[1] Iter[60/227]`
     - `Epoch[1] Iter[80/227]`
     - `Epoch[1] Iter[100/227]`
  3. GPU 已占用约 `8.0GB`，说明训练不是假启动
- 早期观察:
  1. warmup 前段 loss 形状正常：
     - `22.370 -> 17.489`
  2. 目前还未进入 `LPCS` 激活区，因此日志中还看不到 `lpcs_*`
- 当前判断: 继续
- 原因:
  - 当前最关键的节点是 `ep10 / ep20` 和 `epoch 21+` 后 `lpcs_rsr` 是否接近设计预期

### [2026-03-21 22:25] `exp137` 已进入 `LPCS` 激活区，hard-rank 机制按设计生效
- 日志来源:
  - `log/occluded_duke/exp137_lpcs_hard_rank/train_log.txt`
- 新验证点:
  - `ep10 = 36.7 / 50.5`
  - `ep20 = 46.7 / 58.7`
- 对照观察:
  1. 相对 `exp135 ep20 = 46.7 / 58.7`，当前完全重合
  2. 这与设计一致，因为 `exp137` 的唯一改动在 `epoch 21+` 的 `hard-rank LPCS` 聚合，不会影响 warmup 阶段
- 关键机制信号:
  1. `epoch 21+` 后首次稳定出现：
     - `lpcs_rsr = 0.254`
     - `lpcs_psr = 1.000`
     - `lpcs_pf = 1.000`
  2. 这说明当前真正改变的是 `ranking aggregation`，不是 pair routing
  3. `lpcs` 稳定在约 `0.58 ~ 0.64`
  4. `lpcs_dm / lpcs_ds` 已开始从 `0.000` 缓慢抬升到约 `0.004 / 0.001`
  5. `lpcs_fg` 已开始略高于 `lpcs_bg`，当前约为 `0.352 > 0.349`
- 额外观察:
  1. `epoch 21+` 后每轮耗时从约 `58s` 增到约 `72s`
  2. 这符合 hard-top rank selection 引入额外 pair 处理开销的预期，但尚未构成风险
- 当前判断: 继续，当前优先级高
- 原因:
  - 这轮第一次真正把“ranking 对齐而非 routing”单独测起来，而且机制信号与设计完全一致
  - 现在还没有 `ep30`，不能提前判优劣
  - 下一次真正有信息量的节点是 `ep30 / ep40`

### [2026-03-21 13:18] `exp137` 到 `ep60`：hard-rank 机制成立，但当前中期形态偏弱
- 日志来源:
  - `log/occluded_duke/exp137_lpcs_hard_rank/train_log.txt`
- 新验证点:
  - `ep30 = 54.3 / 65.4`
  - `ep40 = 56.5 / 67.7`
  - `ep50 = 57.7 / 68.1`
  - `ep60 = 57.8 / 68.2`
- 对照观察:
  1. 相对 `exp135`：
     - `ep30 = -0.2 mAP / -0.4 R1`
     - `ep40 = -0.2 mAP / -0.6 R1`
     - `ep50 = -0.1 mAP / -1.4 R1`
     - `ep60 = -0.6 mAP / -1.2 R1`
  2. 相对 `exp125 ep60 = 58.0 / 70.6`，当前是 `mAP -0.2 / R1 -2.4`
  3. 相对 `exp030a ep60 = 57.7 / 70.8`，当前是 `mAP +0.1 / R1 -2.6`
- 关键机制信号:
  1. `lpcs_rsr` 全程稳定在 `0.254`，说明 hard-top rank selection 一直处于激活状态
  2. `lpcs_psr / lpcs_pf = 1.000 / 1.000`，说明当前改动确实只打在 ranking aggregation，而不是 pair routing
  3. `lpcs_dm / lpcs_ds` 到 `ep60` 已抬到约 `0.328 / 0.176`
  4. `lpcs_fg` 长期显著高于 `lpcs_bg`，到 `ep60` 约为 `1.294 > 0.577`
- 当前判断: 继续到 `ep80`，但暂不乐观
- 原因:
  - 当前可以确定：hard-rank 不是失效实现，机制完全按设计工作
  - 但到 `ep60` 为止，它没有把 `R1` 拉起来，反而比 full-pair `LPCS` 更弱
  - 鉴于 `pair correction` 类方法此前存在中后期追赶现象，当前仍给它一个 `ep80` 窗口；若到 `ep80` 仍保持 `R1` 明显落后，就应终止这条 hard-rank 支线
