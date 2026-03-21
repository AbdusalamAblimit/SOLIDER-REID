# exp135 监控

## 实验信息
- 方法: `Corrected LPCS Clean Rerun`
- 类型: `exp133` 失效 run 后的共享接线修复重跑
- 运行位置: 待启动（本地）
- 当前状态: 已完成设计建档，待 Claude 审查
- 直接对照:
  - `exp132 LTCS`
  - `exp136 Corrected Sparse LPCS`

## 启动记录

### [2026-03-21 08:58] 设计建档
- 启动原因:
  1. `exp133` 被确认是失效 run，不能用于 LPCS 判断
  2. 当前最紧要的不是切题，而是第一次把真正激活的 `LPCS` 测起来
  3. 本地应优先重跑 corrected `LPCS`，作为 `exp136` 的 clean 直接对照
- 当前判断: 待审查
- 原因:
  - 按用户规则，启动前必须再次通过 Claude 审查

### [2026-03-21 09:02] Claude 审查通过
- 审查文件:
  - `experiments/exp135/claude_review.md`
- 审查结论:
  - **允许启动**
- 审查确认:
  1. 共享接线 bug 已真正修复
  2. `LPCS` loss 会在 `epoch 21+` 真正进入训练
  3. `lpcs_*` 会在日志中出现
  4. 当前相对 intended `exp133` 仅是共享 bug 修复 + 新输出目录，满足 clean rerun 的单变量原则
- 当前判断: 允许启动
- 原因:
  - 这轮终于有资格第一次真正测试 `LPCS`

### [2026-03-21 09:03] 本地 exp135 正式启动
- 启动方式:
  - `/root/miniconda3/envs/solider-reid/bin/python -u train.py --config_file configs/occluded_duke/pose_psg_gcn_lpcs_fix.yml`
- 输出目录:
  - `log/occluded_duke/exp135_lpcs_fix`
- 启动确认:
  1. 日志已确认：
     - `[LPCS] enabled: ... pair_mode=all, top_ratio=1.0 ...`
  2. 已真实进入 iteration：
     - `Epoch[1] Iter[20/227]`
     - `Epoch[1] Iter[40/227]`
     - `Epoch[1] Iter[60/227]`
     - `Epoch[1] Iter[80/227]`
  3. warmup 前段形状与 intended `exp133` 一致，说明修共享接线 bug 没有破坏 baseline 主训练
- 当前判断: 继续
- 原因:
  - 当前最关键的观察点已经明确：`epoch 21+` 后必须首次出现 `lpcs_*`

### [2026-03-21 09:38] `epoch 21+` 已确认 `LPCS` 真正激活
- 日志来源:
  - `log/occluded_duke/exp135_lpcs_fix/train_log.txt`
- 关键验证点:
  - `ep10 = 36.7 / 50.5`
  - `ep20 = 46.7 / 58.7`
  - `ep30 = 54.5 / 65.8`
- 关键机制信号:
  1. `epoch 21+` 后首次稳定出现完整 `lpcs_*`：
     - `lpcs`
     - `lpcs_dm`
     - `lpcs_ds`
     - `lpcs_sm`
     - `lpcs_cm`
     - `lpcs_wm`
     - `lpcs_bg`
     - `lpcs_fg`
     - `lpcs_psr`
     - `lpcs_pf`
  2. `pair_mode=all` 下，`lpcs_psr = 1.000`、`lpcs_pf = 1.000`，与设计一致
  3. `lpcs` 量级稳定在 `0.50 ~ 0.59`，说明不是接上了但几乎没梯度
  4. `lpcs_fg` 持续高于 `lpcs_bg`，到 `ep30` 附近约为 `0.460 > 0.425`
  5. `lpcs_dm / lpcs_ds` 虽小但持续上升，到 `ep30` 已约为 `0.032 / 0.011`
- 对照观察:
  1. 相对 `exp030a ep20 = 46.8 / 60.9`，当前 `exp135 ep20 = 46.7 / 58.7`，表现为 `mAP -0.1 / R1 -2.2`
  2. 相对 `exp125 ep20 = 47.0 / 60.7`，当前也明显偏弱
  3. `ep30 = 54.5 / 65.8` 仍落后于 `exp125 ep30 = 53.4 / 67.1` 的 `R1`
- 当前判断: 继续，但不提前乐观
- 原因:
  - 这轮最重要的任务已经完成：`exp135` 第一次证明修复后的 `LPCS` 确实在训练
  - 但从 `ep20/30` 看，当前 full-pair `LPCS` 早期并不强，下一关键点是 `ep40`
