# exp138 监控

## 实验信息
- 方法: `Rank-Decayed LPCS`
- 类型: `exp135` 的平滑 top-sensitive 单变量升级
- 计划运行位置: 本地
- 当前状态: 训练中
- 直接对照:
  - `exp135 Corrected LPCS`
  - `exp137 Hard-Rank LPCS`

## 启动记录

### [2026-03-21 13:55] 设计建档
- 启动原因:
  1. `exp136` 已证明真稀疏 routing 不是主突破口
  2. `exp137` 已证明 hard-top 25% rank selection 太激进
  3. 因而当前更合理的下一步是：
     - 保留 full-pair 上下文
     - 但用更平滑的方式强调 top-ranked mistakes
- 当前判断: 待审查
- 原因:
  - 按用户规则，训练前必须先完成 Claude 全面审查

### [2026-03-21 14:02] Claude 全面审查已启动
- 审查文件:
  - `experiments/exp138/claude_review.md`
- 审查请求:
  - `experiments/exp138/claude_review_request.txt`
- 审查范围:
  1. `design.md`
  2. config
  3. `defaults.py`
  4. `processor.py`
  5. `pose_backbone_model.py`
  6. `pair_adaptive_fusion.py`
  7. 与 `exp135 / exp137` 的单变量对照关系
- 运行方式:
  - 使用 PTY 会话启动，避免 `nohup` 模式下 `claude` 进程空退出
- 当前判断: 审查进行中，暂不启动训练
- 原因:
  - 用户明确要求先完成全面审查，再由用户告知审查结束

### [2026-03-21 14:24] Claude 全面审查通过，允许启动
- 审查结论:
  - `experiments/exp138/claude_review.md` 明确给出“允许启动”
- 非阻塞提醒:
  1. `tau=8` 在当前 batch 结构下对 `neg` 侧衰减更强，对 `pos` 侧几乎近似均匀
  2. `lpcs_rwm` 更偏向反映 `neg` 侧 rank-decay，不足以单独解释 `pos/neg` 对称性
- 当前判断: 放行，本地启动
- 原因:
  - 没有 blocking 问题，且该实验仍满足相对 `exp135` 的单变量原则

### [2026-03-21 14:20] 正式启动 `exp138`
- 运行方式:
  - 使用本地 `solider-reid` conda 环境启动
  - 当前训练会话: `session_id=46003`
- 启动确认:
  - `[LPCS]` 日志已打印 `rank_mode=rank_decay`
  - `context_mode=none`
  - `POSE_TEST_FEAT=cvk_residual`
- warmup 早期形状:
  - `Epoch[1] Iter[100/227] Loss: 17.489`
  - `Epoch 1 done. Time per epoch: 60.060[s]`
  - `Epoch 2 done. Time per epoch: 58.534[s]`
- 当前判断: 继续
- 原因:
  - 训练健康启动，当前还未进入 `LPCS` warmup 结束后的有效判别区

### [2026-03-21 14:25] warmup 前段运行健康
- 当前进度:
  - `Epoch 4 done. Time per epoch: 58.506[s]`
  - `Epoch 5 done. Time per epoch: 58.581[s]`
  - 当前已进入 `Epoch 6`
- 关键训练日志:
  - `Epoch[5] Iter[200/227] Loss: 7.665, Acc: 0.221`
  - `Epoch[6] Iter[40/227] Loss: 7.534, Acc: 0.127`
- 当前判断: 继续
- 原因:
  - loss 持续下降、acc 正常抬升，当前没有异常发散或实现退化迹象；下一关键点仍是 `ep10`

### [2026-03-21 14:31] `ep10` 首个验证点
- 当前结果:
  - `ep10 = 36.7 / 50.5`
- 对照:
  - `exp135 ep10 = 36.7 / 50.5`
  - `exp137 ep10 = 36.7 / 50.5`
- 当前判断: 继续
- 原因:
  - 这说明 `rank_decay` 在 `epoch<=20` 的 warmup 区间没有扰动主训练形状；真正有信息量的仍是 `epoch 21+` 后 `lpcs_rwm` 是否显著小于 `1.0`

### [2026-03-21 14:48] `exp138` 已越过 warmup，rank-decay 机制明确激活
- 新验证点:
  - `ep20 = 46.7 / 58.7`
- 对照:
  - `exp135 ep20 = 46.7 / 58.7`
  - `exp030a ep20 = 46.8 / 60.9`
- 关键机制信号:
  1. `epoch 21+` 后 `lpcs_*` 已稳定出现
  2. `lpcs_rwm = 0.177`
  3. `lpcs_psr / lpcs_pf / lpcs_rsr = 1.000 / 1.000 / 1.000`
  4. `lpcs_fg > lpcs_bg`，到 `epoch 25` 约为 `0.362 > 0.355`
- 当前判断: 继续，当前为“机制成立、指标仍待拉开”的形态
- 原因:
  - `rank_decay` 已经明确不是空转，且其核心统计 `lpcs_rwm` 显著小于 `1.0`
  - 但到 `ep20` 为止验证仍与 `exp135` 完全重合，下一关键点是 `ep30/40`

### [2026-03-21 14:58] `exp138` 到 `ep30`：机制稳定，但指标仍基本贴着 `exp135`
- 新验证点:
  - `ep30 = 54.4 / 65.8`
- 对照:
  - `exp135 ep30 = 54.5 / 65.8`
  - `exp137 ep30 = 54.3 / 65.4`
  - `exp030a ep30 = 52.2 / 66.0`
- 关键机制信号:
  1. `lpcs_rwm` 持续稳定在 `0.177`
  2. `lpcs_psr / lpcs_pf / lpcs_rsr = 1.000 / 1.000 / 1.000`
  3. `lpcs_fg > lpcs_bg`，到 `epoch 31` 约为 `0.484 > 0.450`
  4. `lpcs_dm / lpcs_ds` 已升到约 `0.030 / 0.010`
- 当前判断: 继续，但优先级下调为“看 `ep40` 是否还能拉开”
- 原因:
  - 现在已经能确认 `rank-decay` 的确在工作，而且比 `hard-top` 更稳
  - 但到 `ep30` 为止，它和 `exp135` 几乎完全等价，还没有出现足够清晰的优势
