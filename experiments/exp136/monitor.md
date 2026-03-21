# exp136 监控

## 实验信息
- 方法: `Corrected Changed-Pair Sparse LPCS`
- 类型: `exp134` 失效 run 后的共享接线修复重跑
- 运行位置: 待启动（远程）
- 当前状态: 已完成设计建档，待 Claude 审查
- 直接对照:
  - `exp135 Corrected LPCS`

## 启动记录

### [2026-03-21 08:58] 设计建档
- 启动原因:
  1. `exp134` 被确认是失效 run，当前还没有真正测到 sparse `LPCS`
  2. 在修复共享接线 bug 后，远程应并行验证 corrected sparse `LPCS`
  3. 这样本地/远程就能形成 clean 的 `full vs sparse` 直接对照
- 当前判断: 待审查
- 原因:
  - 按用户规则，启动前必须再次通过 Claude 审查

### [2026-03-21 09:01] Claude 审查通过
- 审查文件:
  - `experiments/exp136/claude_review.md`
- 审查结论:
  - **允许启动**
- 审查确认:
  1. 共享接线 bug 已真正修复
  2. 相对 `exp135` 仅新增：
     - `POSE_LPCS_PAIR_MODE='delta_top'`
     - `POSE_LPCS_PAIR_TOP_RATIO=0.25`
  3. `lpcs_psr / lpcs_pf` 足以验证 sparse routing 是否真实生效
- 当前判断: 允许启动
- 原因:
  - 已满足远程 clean rerun 条件，可作为 `exp135` 的直接并行对照

### [2026-03-21 09:03] 远程 exp136 正式启动
- 远程机器:
  - 恒源云 `5060 Ti`
- 同步动作:
  1. 本地已 push 到 `origin/exp/pose_heatmap`
  2. 远程已 `git pull origin exp/pose_heatmap`
- 启动命令:
  - `python3 train.py --config_file configs/occluded_duke/pose_psg_gcn_lpcs_delta_top_fix.yml OUTPUT_DIR ./log/occluded_duke/exp136_lpcs_delta_top_fix`
- 远程输出:
  - `log/occluded_duke/exp136_lpcs_delta_top_fix/remote_nohup.log`
- 启动确认:
  1. 日志已确认：
     - `[LPCS] enabled: ... pair_mode=delta_top, top_ratio=0.25 ...`
  2. 已真实进入 iteration：
     - `Epoch[1] Iter[20/227]`
     - `Epoch[1] Iter[40/227]`
  3. 这意味着 corrected sparse `LPCS` 已经正式进入训练，不再是 intended config 或失效 run
- 当前判断: 继续
- 原因:
  - 下一次真正有信息量的节点是 `ep10 / ep20` 和 `epoch 21+` 后的 `lpcs_psr / lpcs_pf`

### [2026-03-21 09:38] corrected sparse `LPCS` 已首次给出有效机制信号
- 日志来源:
  - 远程 `log/occluded_duke/exp136_lpcs_delta_top_fix/remote_nohup.log`
- 关键验证点:
  - `ep10 = 36.4 / 50.1`
  - `ep20 = 47.9 / 59.5`
- 关键机制信号:
  1. `epoch 21+` 后首次稳定出现完整 `lpcs_*`
  2. `lpcs_psr = 0.254`
  3. `lpcs_pf = 2.947 ~ 2.977`
  4. `lpcs_wm = 0.983 ~ 1.000`
  5. `lpcs_dm / lpcs_ds` 目前几乎为 `0.000`
- 机制解释:
  1. 这说明 corrected sparse routing 终于真正接上了，不再是“名义稀疏、实际全开”
  2. `lpcs_psr ≈ 0.25` 与 `top_ratio=0.25` 高度一致，说明当前实现第一次真正测到了 exact 稀疏 pair 路由
  3. `lpcs_pf ≈ 3.0` 也符合“被选中 pair 获得明显放大”的设计预期
- 对照观察:
  1. 相对 `exp135 ep20 = 46.7 / 58.7`，当前 `exp136 ep20 = 47.9 / 59.5`，表现为 `mAP +1.2 / R1 +0.8`
  2. 相对 `exp030a ep20 = 46.8 / 60.9`，当前仍是 `mAP +1.1 / R1 -1.4`
  3. 这更像“排序更积极、R1 仍待观察”的形态，而不是已经明确转强
- 当前判断: 继续，当前优先级高于 `exp135`
- 原因:
  - `exp136` 是第一次真正把 sparse `LPCS` 跑成设计语义
  - 目前最关键的不是 `ep20` 点数本身，而是 `lpcs_psr / lpcs_pf` 已经证明这条机制线终于可以被认真评估
  - 下一次真正有信息量的节点是 `ep30`

### [2026-03-21 10:07] `exp136` 到 `ep30`：机制成立，但指标尚未领先
- 日志来源:
  - 远程 `log/occluded_duke/exp136_lpcs_delta_top_fix/remote_nohup.log`
- 新验证点:
  - `ep30 = 54.5 / 65.7`
- 对照观察:
  1. 相对 `exp135 ep30 = 54.5 / 65.8`，当前几乎完全等价
  2. 相对 `exp030a ep30 = 52.2 / 66.0`，当前是 `mAP +2.3 / R1 -0.3`
  3. 相对 `exp125 ep30 = 53.4 / 67.1`，当前是 `mAP +1.1 / R1 -1.4`
- 关键机制信号:
  1. `lpcs_psr` 持续稳定在 `0.254`
  2. `lpcs_pf` 持续稳定在 `2.88 ~ 3.05`
  3. `lpcs_dm / lpcs_ds` 到 `ep30` 约为 `0.029 / 0.011`
  4. `lpcs_fg` 始终高于 `lpcs_bg`，到 `ep30` 约为 `0.470 > 0.437`
- 当前判断: 继续，先看到 `ep40`
- 原因:
  - 这条线已经完成了最关键的机制验证：真正的 sparse routing 确实在工作
  - 但到 `ep30` 还没有把机制优势兑现成比 `exp135` 更好的指标
  - 当前最需要回答的问题已经收紧为：`sparse LPCS` 是只是“更干净但不更强”，还是会像 `exp125` 一样在中后期出现 late gain
