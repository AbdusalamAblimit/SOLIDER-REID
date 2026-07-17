# 实验 exp142: SKC（Support-Conditioned Keypoint Completion）

## 2026-03-21 19:02 启动前记录

- 状态：仅完成设计，未改代码，未启动训练
- 当前定位：本地主线从 `LPCS` 小变体切换到 feature-space support completion 大改动
- 原因：
  1. `exp109` 已说明真正 headroom 在 `support incomplete`
  2. `exp119-140` 的 pair correction 系列已经提供了足够多“机制有用但突破有限”的证据
  3. 用户明确要求不要继续围绕一个小点耗时间
- 当前判断：
  1. `exp141` 虽已完成二次审查，但仍属于 `LPCS` 家族增量线，暂不启动
  2. `exp142` 将作为本地下一条真正不同的创新点
  3. 下一步先改代码，再做全面 Claude 审查

## 预设监控清单

后续真正启动训练后，每次检查除了常规 `loss / mAP / R1`，还必须补以下行为日志：

- `skc_lmr`
- `skc_spr`
- `skc_arr`
- `skc_gm`
- `skc_gs`
- `skc_dn`
- `skc_pc`
- `skc_pcnt`
- `skc_cl`
- `skc_pre`
- `skc_post`

如果启动后这些日志缺失，则本次 run 视为不可解释 run，需要优先补日志再继续。

## 2026-03-21 19:02 代码接线后自检

- 状态：已完成第一版代码接线，仍未启动训练
- 本轮变更：
  1. 新增 `POSE_SKC` 默认配置与独立 config
  2. 在 `SkeletonGCNHead` 中接入 `Support-Supervised Keypoint Completion`
  3. 在 `processor.py` 中接入：
     - SKC support bank
     - consistency loss
     - 行为日志与 support-target 日志
- 设计修正：
  1. 将最初的 “Support-Conditioned” 收紧为 “Support-Supervised”
  2. 原因是要保证 train/test 一致：
     - 模块本体只依赖当前图
     - support bank 只在训练中作为监督目标
- 自检结果：
  1. `py_compile` 已通过：
     - `model/modules/skeleton_gcn.py`
     - `model/pose_backbone_model.py`
     - `processor/processor.py`
  2. 用 `pose_psg_gcn_skc.yml` 直接构造模型已通过
     - 控制台已打印 `[SKC] Support-Supervised Keypoint Completion enabled`
  3. 最小前向检查已通过：
     - `aux_data` 中已出现
       - `skc_raw_feats`
       - `skc_completed_feats`
       - `skc_scores`
       - `skc_stats`
     - `skc_stats` 已能返回：
       - `low_ratio`
       - `applied_ratio`
       - `gate_mean`
       - `gate_std`
       - `delta_norm`
- 当前判断：
  1. 接线层面已经具备送全面 Claude 审查的条件
  2. 下一步不是启动训练，而是先做广范围审查

## 2026-03-21 19:02 全面 Claude 审查已启动

- 状态：审查中，未启动训练
- 审查方式：
  - 使用 PTY 长会话运行 `claude -p --effort max`
  - 避免后台脱壳造成“看起来启动、实际上没跑”的假状态
- 审查输入：
  - `experiments/exp142/claude_review_request.txt`
- 审查输出目标：
  - `experiments/exp142/claude_review.md`
  - `experiments/exp142/claude_review.err`
- 当前判断：
  1. `exp142` 代码与日志自检已完成
  2. 按用户规则，必须等待 Claude 审查结论后才能启动训练

## 2026-03-21 19:02 第一轮 Claude 审查完成，已按意见修复后准备二审

- 第一轮结论：
  1. 方法方向和代码主体均被放行
  2. 但指出了 1 个中优先级与 2 个低优先级问题
- 第一轮指出的问题：
  1. `applied_ratio` 分母是全部关键点，容易误判
  2. `pre_dist` 纯日志统计却建立了不必要计算图
  3. 缺少 `delta_std`，不利于检测 delta 塌缩
- 已完成修复：
  1. `skc_stats` 新增 `applied_in_low`
  2. `skc_stats` 新增 `delta_std`
  3. `processor.py` 新增对应日志：
     - `skc_ail`
     - `skc_ds`
  4. `raw_norm / pre_dist` 已移入 `torch.no_grad()`
  5. support bank 进度日志改成 warmup 前也可观察
- 修复后自检：
  1. `py_compile` 重新通过
  2. 最小前向已确认 `skc_stats` 键变为：
     - `low_ratio`
     - `applied_ratio`
     - `applied_in_low`
     - `gate_mean`
     - `gate_std`
     - `delta_norm`
     - `delta_std`
- 当前判断：
  1. 现在更适合发起第二轮定向 Claude 审查
  2. 二审通过前，仍不启动训练

## 2026-03-21 19:02 第二轮 Claude 审查通过，允许启动训练

- 二审输出：
  - `experiments/exp142/claude_review_v2.md`
- 核心结论：
  1. 第一轮指出的三项问题均已修复
  2. 当前日志已经足够支撑：
     - 低置信 joints 作用比例
     - 跳过 / 强覆盖判断
     - delta 塌缩判断
     - pre/post support target 距离变化
  3. 无新的 shape / device / dtype / AMP / train-test 对称阻塞项
- 当前判断：
  1. `exp142` 已满足启动条件
  2. 下一步直接进入正式训练与早期监控

## 2026-03-21 19:45 正式训练已启动，warmup 前段健康

- 启动方式：
  1. 前台探针命令成功进入真实训练循环
  2. 因其已稳定进入 iteration，直接保留为本次正式 run
- 输出目录：
  - `log/occluded_duke/exp142_skc`
- 日志文件：
  - `log/occluded_duke/exp142_skc/train_log.txt`
- 当前已确认日志：
  1. `[SKC] enabled: weight=0.5, warmup=20, hidden=256, heads=4, low_thr=0.3, update_thr=0.7, mom=0.9, min_count=1, stop_epoch=-1`
  2. `Epoch[1] Iter[20/227]`
     - `Loss: 22.401`
     - `skc_lmr: 0.146`
     - `skc_arr / skc_ail: 0.000 / 0.000`
     - `skc_spr: 0.027`
     - `skc_pc / skc_pcnt: 0.699 / 2.523`
  3. `Epoch[1] Iter[100/227]`
     - `Loss: 17.688`
     - `skc_lmr: 0.143`
     - `skc_arr / skc_ail: 0.000 / 0.000`
     - `skc_spr: 0.083`
     - `skc_pc / skc_pcnt: 0.844 / 4.634`
- 当前判断：
  1. 启动健康，没有 NaN / OOM / shape 错误
  2. warmup 期间 `skc_arr=0` 是预期行为，因为 `epoch <= 20` 时 `_skc_active=False`
  3. `skc_spr / skc_pc / skc_pcnt` 正在上升，说明 support bank 已在正常积累
  4. 下一关键观察点：
     - `Epoch 1` 结束
     - `ep10`
     - `epoch 21+` 后 `skc_arr / skc_ail / skc_pre / skc_post`

## 2026-03-21 19:50 warmup 前段继续健康，support bank 累积明显加速

- 当前进度:
  - 已完成 `Epoch 1-5`
  - 当前处于 `Epoch 6`
- 关键训练日志:
  1. `Epoch 2 done`
     - `Time per epoch: 59.962[s]`
     - `Speed: 226.3[samples/s]`
  2. `Epoch[3] Iter[200/227]`
     - `Loss: 8.944`
     - `Acc: 0.064`
     - `skc_lmr: 0.147`
     - `skc_arr / skc_ail: 0.000 / 0.000`
     - `skc_spr: 0.147`
     - `skc_pc / skc_pcnt: 0.882 / 40.100`
  3. `Epoch[5] Iter[200/227]`
     - `Loss: 7.723`
     - `Acc: 0.216`
     - `skc_lmr: 0.146`
     - `skc_arr / skc_ail: 0.000 / 0.000`
     - `skc_spr: 0.146`
     - `skc_pc / skc_pcnt: 0.882 / 74.782`
  4. `Epoch[6] Iter[60/227]`
     - `Loss: 7.562`
     - `Acc: 0.139`
     - `skc_lmr: 0.146`
     - `skc_arr / skc_ail: 0.000 / 0.000`
     - `skc_spr: 0.145`
     - `skc_pc / skc_pcnt: 0.881 / 71.918`
- 当前判断: 继续
- 原因:
  1. warmup 形状正常，`Loss` 持续下降，没有 NaN / OOM / shape 异常
  2. `skc_arr / skc_ail / skc_gm / skc_gs / skc_dn / skc_ds` 继续为 `0`，这与 `epoch <= 20` 的设计完全一致
  3. `skc_spr` 已从 `0.027` 抬到约 `0.145`，`skc_pcnt` 已从 `2.523` 抬到 `70+`，说明 support bank 已在稳定积累而不是空转
  4. 下一关键点仍是：
     - `ep10`
     - `ep20`
     - `epoch 21+` 后首次出现的 `skc_arr / skc_ail / skc_cl / skc_pre / skc_post`

## 2026-03-21 20:05 `ep10` 已出，warmup 末端与基线同量级，但 SKC 监督尚未真正激活

- 当前进度:
  - 已完成 `ep10` 验证
  - 当前处于 `Epoch 20`
- 关键验证结果:
  - `ep10 = 38.3 / 51.0`
- 对照观察:
  1. 这与历史 warmup 早期基线同量级，没有出现“新模块一上来就拖坏训练”的信号
  2. `Epoch 10` 训练末端：
     - `Loss: 6.130`
     - `skc_lmr: 0.144`
     - `skc_arr / skc_ail: 0.000 / 0.000`
     - `skc_spr: 0.144`
     - `skc_pc / skc_pcnt: 0.882 / 157.287`
  3. `Epoch 19` 末端：
     - `Loss: 3.388`
     - `Acc: 0.517`
     - `skc_spr: 0.145`
     - `skc_pc / skc_pcnt: 0.882 / 311.168`
  4. `Epoch 20` 前段：
     - `Loss: 3.367 -> 3.403`
     - `skc_arr / skc_ail / skc_gm / skc_gs / skc_dn / skc_ds` 仍全为 `0`
- 当前判断: 继续，真正关键点尚未到来
- 原因:
  1. 现在仍处于 `warmup=20` 的边界内，`_skc_active` 规则是 `epoch > 20`，因此 `Epoch 21` 才是第一次真正测试 SKC completion 的时刻
  2. `skc_spr / skc_pc / skc_pcnt` 已经说明 support bank 准备充分，不存在“激活时 bank 还是空的”问题
  3. 下一次最关键的观察点不是 `ep20` 本身，而是：
     - `Epoch 21` 开始后 `skc_arr / skc_ail` 是否从 `0` 跳起
     - `skc_cl / skc_pre / skc_post` 是否首次出现并呈现 `post < pre`

## 2026-03-21 20:29 `SKC` 已真正激活，但当前更像“稳定小残差修正”而非强 completion

- 当前进度:
  - 已完成 `ep20` 与 `ep40` 验证
  - 当前处于 `Epoch 41`
- 关键验证结果:
  - `ep20 = 47.9 / 60.2`
  - `ep40 = 56.2 / 68.6`
- 关键机制观察:
  1. `Epoch 21` 起 `SKC` 确实首次激活：
     - `skc_arr: 0.154`
     - `skc_ail: 1.000`
     - `skc_cl: 0.136`
     - `skc_pre / skc_post: 0.135 / 0.135`
  2. 到 `Epoch 22` 中段：
     - `skc_dn: 0.096`
     - `skc_ds: 0.013`
     - `skc_gm / skc_gs: 0.120 / 0.000`
  3. 到 `Epoch 39-41`：
     - `skc_arr ≈ 0.140 ~ 0.151`
     - `skc_ail = 1.000`
     - `skc_gm ≈ 0.140 ~ 0.144`
     - `skc_gs ≈ 0.002 ~ 0.003`
     - `skc_dn ≈ 0.67 ~ 0.72`
     - `skc_ds ≈ 0.126 ~ 0.134`
     - `skc_cl ≈ 0.193 ~ 0.199`
     - `skc_pre / skc_post` 仍几乎重合，只在第三位小数有轻微下降
- 当前判断: 继续，但要警惕“弱而均匀的 gate”导致 completion 不够锋利
- 原因:
  1. 正面信号是：
     - `SKC` 不是死路由，激活后所有 low-confidence joints 都真的被改写
     - `delta_norm / delta_std` 持续增大，说明 completion 不是常数零扰动
     - `ep20/40` 至少没有拖垮基线
  2. 目前最值得警惕的点是：
     - `skc_ail=1.0` 说明低置信 joints 几乎全被一刀切处理
     - `skc_gm` 长期只有 `0.12~0.14`，`skc_gs` 又接近 `0`
     - 这更像“近似常数的弱 gate”，而不是按样本/关键点自适应 completion
     - `skc_pre` 到 `skc_post` 的改善仍很弱，说明当前 completion 对 support target 的逼近幅度有限
  3. 现阶段结论应收紧为：
     - `SKC` 已被真正验证到
     - 但第一版还更像”稳定小残差修正”，离强正向 breakthrough 还有距离

## 2026-03-21 20:51 `ep50/ep60` 已出，SKC 跟踪 exp030a 基本持平

- 当前进度:
  - 已完成 `ep50` 与 `ep60` 验证
  - 当前处于 `Epoch 62`
- 关键验证结果:
  - `ep50 = 56.4 / 67.5`
  - `ep60 = 57.8 / 70.3`
- 与 exp030a 同期对照（关键评估）:

| Epoch | exp030a (PSG+GCN) | exp142 (PSG+GCN+SKC) | Δ mAP | Δ R1 |
|-------|-------------------|----------------------|-------|------|
| 10    | 38.2 / 51.3       | 38.3 / 51.0          | +0.1  | -0.3 |
| 20    | 46.8 / 60.9       | 47.9 / 60.2          | +1.1  | -0.7 |
| 30    | 52.2 / 66.0       | 52.0 / 64.3          | -0.2  | -1.7 |
| 40    | 55.6 / 68.6       | 56.2 / 68.6          | +0.6  | ±0.0 |
| 50    | 55.7 / 68.8       | 56.4 / 67.5          | +0.7  | -1.3 |
| 60    | 57.7 / 70.8       | 57.8 / 70.3          | +0.1  | -0.5 |

- 关键机制观察（E55-60 区间）:
  1. `skc_lmr ≈ 0.146`（14.6% 低置信关节点）
  2. `skc_arr ≈ 0.146`，`skc_ail = 1.000`（100% 低置信关节点均被改写）
  3. `skc_gm ≈ 0.17-0.18`，`skc_gs ≈ 0.007-0.009`（gate 增长但仍偏低且方差极小）
  4. `skc_dn ≈ 1.0-1.1`，`skc_ds ≈ 0.20-0.22`（delta 范数非零，有一定变化）
  5. **`skc_pre ≈ 0.205 ≈ skc_post`**（completion 未将 low-conf token 拉向 support target）
  6. `skc_cl ≈ 0.206`（completion loss 稳定但不降）
- 当前判断: **SKC 中性偏负**，继续跑到 E80 再做最终判断
- 核心分析:
  1. **mAP 与 exp030a 完全持平**：6 个验证点中，exp142 的 mAP 波动始终在 exp030a ±1% 以内
  2. **R1 略差**：多数验证点 R1 低于 exp030a，尤其 E30/E50 差 1.3-1.7%
  3. **completion 未真正兑现**：`skc_pre ≈ skc_post` 说明模块虽然在修改特征，但修改方向不是向 support prototype 靠近
  4. **gate 太均匀**：`skc_gs ≈ 0.008` 说明 gate 几乎是常数，未实现 “按样本/关键点自适应” 的设计目标
  5. **exp030a 后半程参考**：exp030a 从 E60(57.7%) 涨到 E120(61.1%)，若 exp142 也涨同样幅度，最终应在 ~61.2%，即与基线持平
- 下一关键判断点:
  - `ep70`：若 mAP < 58.0%，而 exp030a 同期 58.1%，可初步判定中性
  - `ep80`：若 mAP < 59.0%（exp030a E80 = 59.4%），则确认 SKC 无正向收益，应准备止损

## 2026-03-21 21:02 `ep70` 结果积极！exp142 首次拉开与 exp030a 的正差距

- 当前进度:
  - 已完成 `ep70` 验证
  - 当前处于 `Epoch 72+`
- 关键验证结果:
  - `ep70 = 58.7 / 71.2`
- 与 exp030a 同期对照更新:

| Epoch | exp030a (PSG+GCN) | exp142 (PSG+GCN+SKC) | Δ mAP | Δ R1 |
|-------|-------------------|----------------------|-------|------|
| 10    | 38.2 / 51.3       | 38.3 / 51.0          | +0.1  | -0.3 |
| 20    | 46.8 / 60.9       | 47.9 / 60.2          | +1.1  | -0.7 |
| 30    | 52.2 / 66.0       | 52.0 / 64.3          | -0.2  | -1.7 |
| 40    | 55.6 / 68.6       | 56.2 / 68.6          | +0.6  | ±0.0 |
| 50    | 55.7 / 68.8       | 56.4 / 67.5          | +0.7  | -1.3 |
| 60    | 57.7 / 70.8       | 57.8 / 70.3          | +0.1  | -0.5 |
| **70** | **58.1 / 70.9** | **58.7 / 71.2** | **+0.6** | **+0.3** |

- 关键机制观察（E65-70 区间）:
  1. `skc_gm ≈ 0.205`（从 E60 的 0.18 持续增长）
  2. `skc_gs ≈ 0.012`（方差也在增加，gate 开始有更多变化）
  3. `skc_dn ≈ 1.25`（delta 范数从 E60 的 1.05 继续增大）
  4. `skc_ds ≈ 0.25`（delta 标准差也在增大）
  5. `skc_pre/skc_post` 仍接近（E69: 0.201/0.201），但整体水平从 0.205 降到 0.201
- 当前判断: **积极信号**，继续跑
- 核心分析:
  1. E60→E70 的增量：exp142 +0.9% mAP vs exp030a +0.4% mAP，说明 SKC 在后期阶段开始提供额外收益
  2. gate 和 delta 仍在稳定增长，说明 completion 机制在逐步强化（而非塌缩）
  3. 但 `skc_pre ≈ skc_post` 的持续现象说明增益来源不是向 support prototype 靠近，可能是 completion 模块本身作为额外的 feature transformation 提供了正则化效果
  4. **E80 将是决定性验证点**：exp030a E80 = 59.4%，若 exp142 E80 > 59.4%，则可初步确认 SKC 有训练端正收益

## 2026-03-21 21:13 `ep80` 确认 +0.6% mAP 正向信号！

- 当前进度:
  - 已完成 `ep80` 验证
  - 当前处于 `Epoch 82+`
  - ETA: ~40 分钟
- 关键验证结果:
  - `ep80 = 60.0 / 72.0`
- 与 exp030a 同期对照更新:

| Epoch | exp030a (PSG+GCN) | exp142 (PSG+GCN+SKC) | Δ mAP | Δ R1 |
|-------|-------------------|----------------------|-------|------|
| 60    | 57.7 / 70.8       | 57.8 / 70.3          | +0.1  | -0.5 |
| 70    | 58.1 / 70.9       | 58.7 / 71.2          | +0.6  | +0.3 |
| **80** | **59.4 / 72.6** | **60.0 / 72.0** | **+0.6** | **-0.6** |

- 关键机制观察（E78-80）:
  1. `skc_gm = 0.227`（持续增长，gate 在逐步打开）
  2. `skc_gs = 0.017`（方差也在增加，gate 越来越有样本差异性）
  3. `skc_dn = 1.34`（delta 范数大幅增长）
  4. `skc_ds = 0.283`（delta 标准差也显著增大）
  5. `skc_cl = 0.204`（completion loss 缓慢下降）
  6. `skc_pre/skc_post`: 0.203/0.202（极微弱的正向差距开始出现）
- 当前判断: **mAP 正信号已连续 3 个验证点确认**（E60 +0.1%, E70 +0.6%, E80 +0.6%）
- 核心分析:
  1. **mAP +0.6% 正信号稳定**：E70 和 E80 都保持 +0.6%，趋势一致
  2. **R1 波动大**：E70 +0.3% 但 E80 -0.6%，R1 不稳定
  3. **completion 机制在持续强化**：gate 从 0.18(E60) → 0.22(E80)，delta norm 从 1.05 → 1.34
  4. **exp030a E120 单 seed = 61.1%**：若 exp142 维持 +0.6% 优势，最终预期 ~61.7%
  5. **exp030a-eq 3-seed mean = 60.73%**：exp142 若到 61.7%，相对 3-seed mean 约 +1.0%
  6. ⚠️ **重要修正**：E80 对照发现 exp142 用 `equal_concat` eval，而 exp030a 用 `concat_scaled` eval。两者 mode 不同导致之前的 +0.6% 对比有误。修正后：
     - exp030a `concat_scaled` E80 = 59.4%，E120 final = 60.5%
     - exp030a `equal_concat` E120 final = 61.1%，mode 差 ≈ +0.6%
     - 估算 exp030a `equal_concat` E80 ≈ 60.0%
     - exp142 `equal_concat` E80 = 60.0%，**完全持平**
  7. 因此 SKC 大概率是**中性**，而非之前预期的正信号
  8. 最终 E120 结果将给出定论
- 下一步:
  - 继续让训练跑完 E120
  - E90/E100/E110 持续跟踪
  - E120 最终结果决定是否值得多 seed 验证

## 2026-03-21 21:24 `ep90` 出现回落，SKC 后期可能干扰

- 当前进度:
  - 已完成 `ep90` 验证
  - 当前处于 `Epoch 92+`
- 关键验证结果:
  - `ep90 = 59.4 / 71.3`
- 完整训练曲线对照:

| Epoch | exp030a concat_scaled | exp030a est. equal_concat | exp142 equal_concat | Δ vs est. |
|-------|----------------------|--------------------------|---------------------|-----------|
| 80    | 59.4%                | ≈60.0%                    | 60.0%               | ±0.0      |
| 90    | 60.2%                | ≈60.8%                    | **59.4%**           | **-1.4%** |

- 当前判断: **E90 信号转负**
- 核心分析:
  1. exp142 从 E80(60.0%) 到 E90(59.4%) 下降了 0.6%
  2. 同期 exp030a 从 E80(59.4%) 到 E90(60.2%) 上升了 0.8%（concat_scaled mode）
  3. 这意味着 exp142 在 E80-E90 区间的表现比 exp030a 差约 1.4%
  4. 可能原因：
     - SKC 的 completion loss 在后期产生了梯度干扰
     - gate 持续增大（0.239 at E86），可能 delta 过大导致 keypoint 特征偏离
     - support bank 在后期可能积累了噪声
  5. 但 E80-E90 的 dip 也可能是暂时波动，exp066(PAA) 也有过 E80→E90 的小幅回落
- 下一步:
  - 等待 E100, E110, E120 判断是回落还是暂时波动
  - 若 E100 < 60.0%，基本可确认 SKC 中性偏负
  - 若 E100 ≥ 60.5%，说明 E90 是暂时波动

## 2026-03-21 21:35 `ep100` 部分恢复但仍落后于 exp030a

- 关键验证结果:
  - `ep100 = 59.9 / 71.4`
- 完整后半程轨迹:

| Epoch | exp030a cs | exp030a est. eq | exp142 eq | Δ vs est. |
|-------|-----------|----------------|-----------|-----------|
| 80    | 59.4%     | ≈60.0%          | 60.0%     | ±0.0      |
| 90    | 60.2%     | ≈60.8%          | 59.4%     | -1.4%     |
| 100   | 60.1%     | ≈60.7%          | 59.9%     | -0.8%     |

- 当前判断: **SKC 中性偏负**
- 核心分析:
  1. E100 回到 59.9%（比 E90 的 59.4% 恢复 +0.5%），但仍低于 E80 的 60.0%
  2. exp030a 在 E100 的 concat_scaled 是 60.1%，估算 equal_concat ≈ 60.7%
  3. exp142 落后约 0.8%
  4. SKC 的 gate 仍在无限制增长（0.254 at E100），delta norm 1.47
  5. 这个不受控增长可能是后期性能下滑的原因
  6. **结论：SKC 的核心问题是 gate/delta 增长没有上界，导致后期过度修改特征**
- exp030a final eq: 61.1% (单 seed)。若 exp142 延续当前趋势，最终预计 60.0-60.5%，约 -0.6~-1.1% vs exp030a

## 2026-03-21 21:57 训练完成：SKC 确认为中性偏负

- **最终结果**:
  - `E120 = 60.3% mAP / 71.8% R1 / 84.4% R5 / 87.7% R10`（使用 `equal_concat` 模式）
- 与 exp030a 对照:
  - exp030a `equal_concat` E120 = **61.1%** mAP / **73.7%** R1（单 seed）
  - Δ = **-0.8% mAP / -1.9% R1**
  - exp030a 3-seed mean = **60.73%** mAP / **72.57%** R1
  - vs 3-seed mean: Δ = **-0.43% mAP / -0.77% R1**
- 完整训练曲线（equal_concat 模式）:

| Epoch | exp142 SKC | exp030a (est. eq) | Δ |
|-------|-----------|-------------------|---|
| 60    | 57.8%     | ≈58.3%             | -0.5 |
| 70    | 58.7%     | ≈58.7%             | 0.0 |
| 80    | 60.0%     | ≈60.0%             | 0.0 |
| 90    | 59.4%     | ≈60.8%             | -1.4 |
| 100   | 59.9%     | ≈60.7%             | -0.8 |
| 110   | 60.3%     | ≈61.0%             | -0.7 |
| **120** | **60.3%** | **≈61.1%**       | **-0.8** |

- **结论**:
  1. SKC 是**中性偏负**结果
  2. 在 E60-E80 阶段曾短暂与 exp030a 持平，但 E90 开始出现回落
  3. 最终 -0.8% mAP / -1.9% R1，SKC 没有为 ReID 带来正向收益
  4. **核心失败原因分析**:
     - `skc_pre ≈ skc_post`：completion 模块虽然在修改特征（delta norm ≈ 1.5），但修改方向不是向 support prototype 靠近
     - gate 无限制增长（0.12 → 0.26），后期过度修改了本来已经有效的 keypoint 特征
     - completion consistency loss (skc_cl ≈ 0.20) 未能有效引导模块行为
     - 本质上，SKC 的 "support-supervised completion" 思路在 15K 数据集上不成立——模型无法学会从 support bank 中提取有用的 completion 信号
  5. **对后续方向的启示**:
     - feature-level completion 方向已被多种方式验证为负/中性（SGMKC, SCRC, SCKD, SKC）
     - 应放弃"在特征层修复缺失证据"的思路
     - 转向更轻量的注意力 inductive bias 方向（如 SASA）
