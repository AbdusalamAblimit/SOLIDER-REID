# exp132 监控

## 实验信息
- 方法: LTCS（Learn-to-Trust Common Support）
- 类型: 新主线 / learned pair-adaptive fusion
- 运行位置: 待定（优先本地 3090）
- 主配置: `configs/occluded_duke/pose_psg_gcn_ltcs.yml`
- 核心变量: 相对 `exp030a` 新增真正接入检索的 pair-adaptive fusion head
- 直接对照:
  - `exp030a-eq seed1234`
  - 固定 `cvk_hybrid`
  - `exp125`

## 启动记录

### [2026-03-21 02:35] 设计建档
- 启动原因:
  1. `exp130` 已较干净地否定 `target form` 是主瓶颈
  2. `exp131` 已较干净地否定 `relation coverage` 是主瓶颈
  3. 当前最合理的新假设是：
     - pair-specific correction 不能继续被压进单个 embedding
     - 应把 correction rule 本身学出来，并真正接入检索
- 当前判断: 待实现
- 原因:
  - 先完成代码实现与 Claude 审查，再允许启动训练

### [2026-03-21 04:05] 代码实现完成，等待审查放行
- 已完成:
  1. 新增 `model/modules/pair_adaptive_fusion.py`
  2. `PoseBackboneModel` 已注册 `ltcs_head`，并支持 `POSE_TEST_FEAT='cvk_adaptive'`
  3. `processor.py` 已接入：
     - LTCS teacher bank
     - 训练期 `ltcs_loss`
     - 验证/测试期 `evaluator.pair_fusion_head`
  4. `utils/metrics.py` 已接入 `cvk_adaptive` 评估路径，且按 query chunk 推理，避免一次性堆满 `Q×G×D`
  5. 配置已建：
     - `configs/occluded_duke/pose_psg_gcn_ltcs.yml`
- 已通过的自检:
  1. `py_compile` 通过
  2. 使用 `solider-reid` 环境做了最小 evaluator 样例自检，`cvk_adaptive` 路径可正常跑通
  3. `LTCS` 头初始化已改为近似 `alpha=0.5`，避免 warmup 早期被随机融合扰动
- 当前判断: 等待 Claude 审查
- 原因:
  - 按用户明确规则，所有新实验必须先经过 Claude 审查后才能启动训练

### [2026-03-21 04:15] Claude 审查状态
- 审查文件:
  - `experiments/exp132/claude_review.md`
- 当前状态:
  1. 已通过本地 `claude -p` 启动针对 `exp132` 代码接线的专项审查
  2. 审查重点已收缩到真正影响启动的部分：
     - head 是否进 checkpoint
     - evaluator/test 是否真调用
     - `cvk_adaptive` 是否存在明显设备/内存风险
  3. 目前审查仍在运行，尚未产出最终落盘内容
- 当前判断: 暂不启动训练
- 原因:
  - 不越过用户设定的“先审查、后启动”规则

### [2026-03-21 04:18] Claude 审查结论
- 审查文件:
  - `experiments/exp132/claude_review.md`
- 审查结论:
  1. **允许启动**
  2. 无 HIGH 阻塞项
  3. 唯一需要明确接受的点是：
     - 当前 `LTCS` loss 只更新 `ltcs_head`
     - 不会反向塑造 backbone / GCN
     - 这与当前设计目标一致：先验证 learned pair-adaptive fusion rule 本身
- 三项关键核查:
  1. `ltcs_head` 已属于模型参数，`state_dict()` 会保存
  2. `cvk_adaptive` 测试路径已真正调用 `pair_fusion_head`
  3. 当前实现按 query chunk 评估，未发现明显设备错配或内存风险
- 当前判断: 允许启动
- 原因:
  - 审查已经明确放行；下一步进入最小启动自检与正式训练

### [2026-03-21 04:19] 启动前最小自检通过
- 启动方式:
  - `/root/miniconda3/envs/solider-reid/bin/python -u train.py --config_file configs/occluded_duke/pose_psg_gcn_ltcs.yml`
- 关键确认:
  1. 输出目录正确：`log/occluded_duke/exp132_ltcs`
  2. `POSE_TEST_FEAT=cvk_adaptive` 已生效
  3. `LTCS` 模块已注册进模型：
     - `[LTCS] Learn-to-Trust Common Support enabled: hidden=32, params=1313`
  4. 训练期日志已确认：
     - `[LTCS] enabled: weight=0.5, warmup=20, hidden=32, low_thr=0.3, update_thr=0.7, mom=0.9, min_count=1, stop_epoch=-1`
  5. 数据集、优化器、模型构建全部正常，没有接线错误或启动期报错
- 当前判断: 允许正式运行
- 原因:
  - 该自检已经覆盖了最关键的配置/接线风险；下一步转为后台正式训练即可

### [2026-03-21 04:21] 正式后台训练启动
- 处理动作:
  1. 终止仅用于启动自检的前台 run，避免后续监控与正式实验混淆
  2. 保留部分日志到：
     - `log/occluded_duke/exp132_ltcs_pre_restart1/train_log.txt`
  3. 使用 `setsid + bash -lc` 方式重启为正式后台训练
- 当前正式进程:
  - 主进程 PID: `496117`
  - 输出目录: `log/occluded_duke/exp132_ltcs`
  - 后台日志: `log/occluded_duke/exp132_ltcs/nohup.log`
- 启动确认:
  1. `nohup.log` 已正常落盘
  2. `train_log.txt` 已重新生成
  3. 训练进程及 DataLoader worker 均已出现，说明后台运行稳定
- 当前判断: 继续
- 原因:
  - Claude 审查已放行，正式 run 已按独立后台流程启动成功；下一步进入常规监控
