# exp133 监控

## 实验信息
- 方法: `LPCS`（Learned Pair Correction Scorer）
- 类型: `exp132` 之后的新主线候选
- 运行位置: 未启动
- 当前状态: 仅完成设计建档，等待实现与 Claude 审查
- 直接对照:
  - `exp132a cvk_adaptive`
  - `exp132b cvk_hybrid`
  - `exp125`

## 启动记录

### [2026-03-21 07:25] 设计建档
- 启动原因:
  1. `exp132` 已较干净地否定第一版 `alpha-fusion`
  2. learned pair module 的大方向仍未被否定
  3. 当前最合理的升级是：
     - 从“学该信谁”
     - 转到“学该修正多少”
- 当前判断: 待实现
- 原因:
  - 先完成设计收束，再按用户规则做 Claude 审查，审查通过后才允许启动

### [2026-03-21 07:34] 代码实现完成，等待 Claude 审查
- 已完成:
  1. 在 `model/modules/pair_adaptive_fusion.py` 新增 `PairResidualScorer`
  2. `PoseBackboneModel` 已注册 `lpcs_head`
  3. `processor.py` 已接入：
     - `LPCS` support-complete teacher bank
     - teacher-weighted pairwise ranking loss
     - 验证/测试期 `pair_residual_head`
  4. `utils/metrics.py` 已接入 `POSE_TEST_FEAT='cvk_residual'`
  5. 配置已建：
     - `configs/occluded_duke/pose_psg_gcn_lpcs.yml`
- 当前这版的机制定义:
  1. 保持 `exp132` 的 pair descriptor 不变
  2. 不再预测 `alpha`
  3. 而是预测 bounded `delta`
  4. 最终距离:
     - `d_final = d_cvk_hybrid + delta`
  5. 监督改为：
     - support-complete teacher 加权的 pairwise ranking loss
- 已通过的自检:
  1. `py_compile` 通过
  2. 最小 evaluator 样例已确认 `cvk_residual` 路径可正常跑通
- 当前判断: 等待 Claude 审查
- 原因:
  - 按用户明确规则，所有新实验必须先完成 Claude 审查并确认无阻塞项后才能启动

### [2026-03-21 08:06] Claude 审查结论
- 审查文件:
  - `experiments/exp133/claude_review.md`
- 审查结论:
  1. **允许启动**
  2. 无 HIGH 阻塞项
  3. 两个 MEDIUM 问题均不阻止当前实验启动：
     - 训练端 `base_dist` 与测试端 `CVK_GLOBAL_WEIGHT / CVK_KP_WEIGHT` 的耦合暂时是硬编码 `1:1`
     - warmup 期间 `LPCS` head 仅受极小 weight decay 影响
- 当前接受的边界:
  1. 当前实验配置下 `CVK_GLOBAL_WEIGHT = CVK_KP_WEIGHT = 1.0`，所以 train-test base distance 一致
  2. 本轮不再修改代码以避免越过已完成的 Claude 审查；后续若这条线成立，再统一到配置驱动
- 当前判断: 允许启动
- 原因:
  - Claude 已确认单变量性、checkpoint 接线、train loss 接线、test/evaluator 接线和 ranking loss 方向均成立

### [2026-03-21 08:07] 启动前最小自检通过
- 启动方式:
  - `/root/miniconda3/envs/solider-reid/bin/python -u train.py --config_file configs/occluded_duke/pose_psg_gcn_lpcs.yml`
- 关键确认:
  1. 输出目录正确：`log/occluded_duke/exp133_lpcs`
  2. `POSE_TEST_FEAT=cvk_residual` 已生效
  3. `LPCS` 模块已注册进模型：
     - `[LPCS] Learned Pair Correction Scorer enabled: hidden=32, delta_scale=0.5, params=1313`
  4. 训练期日志已确认：
     - `[LPCS] enabled: weight=0.5, warmup=20, hidden=32, delta_scale=0.5, low_thr=0.3, update_thr=0.7, mom=0.9, min_count=1, stop_epoch=-1`
  5. 训练已真实进入 iteration：
     - `Epoch[1] Iter[60/227]` 正常输出，无 NaN / shape / device 报错
- 当前判断: 允许正式运行
- 原因:
  - Claude 审查已放行，且前台最小自检已证明 `LPCS` 不只是“能构建”，而是能真正进入训练循环

### [2026-03-21 08:09] 正式后台训练启动
- 处理动作:
  1. 终止仅用于启动自检的前台 run，避免后续监控与正式实验混淆
  2. 保留部分日志到：
     - `log/occluded_duke/exp133_lpcs_pre_restart1/train_log.txt`
  3. 使用 `setsid + bash -lc` 方式重启为正式后台训练
- 当前正式进程:
  - 主进程 PID: `751830`
  - 输出目录: `log/occluded_duke/exp133_lpcs`
  - 后台日志: `log/occluded_duke/exp133_lpcs/nohup.log`
- 启动确认:
  1. `nohup.log` 已正常落盘
  2. `train_log.txt` 已重新生成
  3. 训练进程及 DataLoader worker 均已出现，说明后台运行稳定
- 当前判断: 继续
- 原因:
  - `exp133` 已经进入正式训练阶段；下一步按早期监控节奏检查 warmup 是否与 `exp132` 同量级稳定

### [2026-03-21 08:10] warmup 早期检查
- 当前进度:
  - 主进程 PID: `751830`
  - 当前已到 `Epoch[1] Iter[160/227]`
- 当前日志:
  - `Iter20  Loss 22.370`
  - `Iter40  Loss 20.187`
  - `Iter60  Loss 18.990`
  - `Iter80  Loss 18.161`
  - `Iter100 Loss 17.489`
  - `Iter120 Loss 17.020`
  - `Iter140 Loss 16.581`
  - `Iter160 Loss 16.189`
- 当前观察:
  1. 与 `exp132` 前段 warmup 形状基本一致，说明 `LPCS` 没有在 `epoch<=20` 阶段引入额外副作用
  2. 这也符合当前设计：`LPCS` warmup 前只是在后台积累 support teacher bank，不会提前干扰 backbone 主训练
  3. 目前尚未进入 `LPCS` 激活阶段，因此现在只能确认“启动健康”，不能提前判断有效性
- 当前判断: 继续
- 原因:
  - 早期训练健康，下一次真正有信息量的节点是 `ep10 / ep20` 和 `epoch 21+` 后的 `lpcs_*` 统计

### [2026-03-21 08:13] 首个验证点
- 当前进度:
  - `Epoch 10` 验证已完成
- 验证结果:
  - `mAP: 36.7%`
  - `Rank-1: 50.5%`
- 当前观察:
  1. 这一形状与 `exp132` 的早期 `36.7 / 50.5` 基本一致
  2. 说明 `LPCS` 在 warmup 阶段没有拖坏主训练
  3. 当前仍未进入 `epoch 21+` 的 `LPCS` 真正激活区，所以还不能据此判断 learned pair correction 是否成立
- 当前判断: 继续
- 原因:
  - 这是一个健康的基线对齐信号，不是有效性结论；真正关键的是 `ep20` 之后的 `lpcs_*` 统计和 `ep30/40` 验证
