# exp137 监控

## 实验信息
- 方法: `Hard-Rank LPCS`
- 类型: `exp135` 的 ranking-aligned 单变量升级
- 运行位置: 待启动（本地）
- 当前状态: 已完成设计建档，待代码接线自检与 Claude 审查
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
