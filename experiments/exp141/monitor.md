# exp141 监控

## 实验信息
- 方法: `Competition-Context LPCS`
- 类型: `exp135` 的 context 单变量升级
- 计划运行位置: 远程
- 当前状态: 远程训练中
- 直接对照:
  - `exp135 Corrected LPCS`
  - `exp139 Query-Context LPCS`

## 启动记录

### [2026-03-22 02:45] 设计建档并接线完成，等待全面 Claude 审查
- 启动原因:
  1. `exp139` 说明 query-level context 确实有效
  2. 但到 `ep100` 为止，它仍更像“稳住主候选”，还没有形成明显超越
  3. 当前更值得测的是：
     - 真正重要的是否不是 query 均值摘要
     - 而是 **当前 candidate 在本 query 全部候选里的相对竞争位置**
- 核心改动:
  1. 为 `LPCS` 新增 `POSE_LPCS_CONTEXT_MODE='comp_ctx'`
  2. 新增 5 维 pair-specific competition context：
     - `base_rank`
     - `kp_rank`
     - `support_rank`
     - `gain_rank`
     - `gain_zscore`
  3. 训练与测试都按 query 的 candidate set 直接构造，无标签、train/test 对称
- 当前判断: 待审查
- 原因:
  - 用户要求新实验先走全面 Claude 审查，再由用户告知审查结束后启动

### [2026-03-22 02:49] 本地自检通过，准备发起全面 Claude 审查
- 自检结果:
  1. `py_compile` 已通过：
     - `model/modules/pair_adaptive_fusion.py`
     - `model/pose_backbone_model.py`
     - `processor/processor.py`
     - `utils/metrics.py`
  2. competition context 最小样例检查通过：
     - `base_desc = [4, 4, 6]`
     - `comp_ctx = [4, 4, 5]`
     - `concat = [4, 4, 11]`
  3. `comp_ctx.abs().mean() > 0`
     - 说明不是全零接线
- 当前判断: 可以送审，但暂不启动训练
- 原因:
  - 现在已经满足“全面审查前的最小自检”要求，下一步只等 Claude 审查结论

### [2026-03-22 02:55] 首轮 Claude 审查未通过，已确认是 config 模板错误
- 审查文件:
  - `experiments/exp141/claude_review.md`
- blocking 结论:
  1. 当前 `pose_psg_gcn_lpcs_comp_ctx.yml` 不是从 `exp135` config 严格继承
  2. 至少存在多处关键字段差异，会破坏单变量原则
  3. 其中包含 `MODEL.NAME` 等高风险项，不能放行
- 当前处理:
  1. 按审查建议，将 `exp141` config 改成严格复制 `exp135`
  2. 仅保留两处差异：
     - `POSE_LPCS_CONTEXT_MODE: 'comp_ctx'`
     - `OUTPUT_DIR`
- 当前判断: 修复完成，准备二次全面审查
- 原因:
  - 审查否掉的不是 `comp_ctx` 机制本身，而是实验隔离性不干净

### [2026-03-22 02:57] 二次自检通过，已满足重新送审条件
- 自检结果:
  1. 当前 config 相对 `exp135` 的 diff 只剩两处：
     - `POSE_LPCS_CONTEXT_MODE: 'comp_ctx'`
     - `OUTPUT_DIR`
  2. 这意味着 `exp141` 现在已满足单变量前提
- 审查文件:
  - 请求: `experiments/exp141/claude_review_request_v2.txt`
  - 输出: `experiments/exp141/claude_review_v2.md`
- 当前判断: 等待二次全面审查，不启动训练
- 原因:
  - 用户要求由用户确认审查结束后再继续

### [2026-03-22 03:51] 二次审查通过后改为远程正式启动
- 启动原因:
  1. 本地主线已切到 `exp142 SKC`，需要把真正不同机制的第二创新点放到远程并行验证
  2. `exp141` 与 `exp142` 分别代表：
     - `exp141`: retrieval-side competition context
     - `exp142`: feature-space support-supervised completion
  3. `exp141` 二次全面审查已明确放行：
     - `experiments/exp141/claude_review_v2.md`
- 同步动作:
  1. 远程旧代码缺少 `pose_psg_gcn_lpcs_comp_ctx.yml`
  2. 已从本地精确同步以下文件到远程 `/root/work/SOLIDER-REID`：
     - `config/defaults.py`
     - `model/pose_backbone_model.py`
     - `model/modules/pair_adaptive_fusion.py`
     - `model/modules/skeleton_gcn.py`
     - `processor/processor.py`
     - `utils/metrics.py`
     - `configs/occluded_duke/pose_psg_gcn_lpcs_comp_ctx.yml`
     - `experiments/exp141/*`
- 启动命令:
  - `python3 train.py --config_file configs/occluded_duke/pose_psg_gcn_lpcs_comp_ctx.yml OUTPUT_DIR ./log/occluded_duke/exp141_lpcs_comp_ctx`
- 远程输出:
  - `log/occluded_duke/exp141_lpcs_comp_ctx/remote_nohup.log`
- 启动确认:
  1. 远程主训练进程已存在：
     - `python3 train.py --config_file configs/occluded_duke/pose_psg_gcn_lpcs_comp_ctx.yml OUTPUT_DIR ./log/occluded_duke/exp141_lpcs_comp_ctx`
  2. 远程 GPU 已被占用：
     - `5060 Ti` 显存约 `6684 MiB`
  3. 远程配置日志已确认：
     - `POSE_LPCS_CONTEXT_MODE: comp_ctx`
     - `POSE_TEST_FEAT: cvk_residual`
- 当前判断: 继续
- 原因:
  - `exp141` 现在已经作为和 `exp142` 明确不同的第二主线被干净放到远程验证

### [2026-03-22 03:52] warmup 前段运行健康，已进入稳定 iteration
- 当前进度:
  - 已完成 `Epoch 1` 过半
  - 当前达到 `Epoch[1] Iter[140/227]`
- 关键训练日志:
  1. `Epoch[1] Iter[20/227]`
     - `Loss: 22.716`
     - `Acc: 0.002`
  2. `Epoch[1] Iter[100/227]`
     - `Loss: 18.041`
     - `Acc: 0.002`
  3. `Epoch[1] Iter[140/227]`
     - `Loss: 16.955`
     - `Acc: 0.002`
- 当前判断: 继续
- 原因:
  1. warmup 形状正常，`Loss` 稳定下降，没有启动异常
  2. `LPCS` 仍处于 `warmup=20` 阶段，现在只能判断“启动健康”，还不能对 `comp_ctx` 本身下结论
  3. 下一关键观察点：
     - `Epoch 1` 结束
     - `ep10`
     - `epoch 21+` 后首次出现的 `lpcs_ctxm / lpcs_fg / lpcs_bg / lpcs_dm / lpcs_ds`

### [2026-03-22 04:05] 远程 warmup 持续健康，已推进到 `Epoch 9`
- 当前进度:
  - 已完成 `Epoch 1-9`
  - 尚未到首个验证点 `ep10`
- 关键训练日志:
  1. `Epoch 2 done`
     - `Time per epoch: 91.547[s]`
     - `Speed: 148.2[samples/s]`
  2. `Epoch 5 done`
     - `Time per epoch: 91.194[s]`
     - `Speed: 148.8[samples/s]`
  3. `Epoch 9` 末端：
     - `Epoch[9] Iter[200/227] Loss: 6.288`
     - `Acc: 0.231`
  4. 训练曲线整体：
     - `Epoch 3 Iter[20] Loss: 9.757`
     - `Epoch 6 Iter[200] Loss: 7.319`
     - `Epoch 9 Iter[200] Loss: 6.288`
- 当前判断: 继续
- 原因:
  1. warmup 形状与 `query_ctx` 线相近，没有发现数值异常
  2. 目前仍无法对 `comp_ctx` 的机制价值下判断，因为 `LPCS warmup=20`
  3. 下一关键点是：
     - `ep10`
     - `epoch 21+` 后是否首次出现并稳定抬起 `lpcs_ctxm`

### [2026-03-22 04:29] `comp_ctx` 已过 warmup，机制成功接上，但目前还没显示出超出 `query_ctx` 的明显优势

- 当前进度:
  - 已完成 `ep20` 验证
  - 当前处于 `Epoch 22`
- 关键验证结果:
  - `ep20 = 47.6 / 60.0`
- 关键机制观察:
  1. `Epoch 21` 起 `LPCS` 真正激活：
     - `lpcs: 0.604 -> 0.577`
     - `lpcs_ctxm: 0.543 ~ 0.544`
     - `lpcs_psr / lpcs_pf / lpcs_rsr / lpcs_rwm = 1.000 / 1.000 / 1.000 / 1.000`
  2. `Epoch 21` 内：
     - `lpcs_bg: 0.275 -> 0.314`
     - `lpcs_fg: 0.276 -> 0.314`
     - `lpcs_dm / lpcs_rdm: -0.000 -> -0.009`
     - `lpcs_ds: 0.000 -> 0.010`
  3. `Epoch 22` 前半：
     - `lpcs: 0.542 ~ 0.558`
     - `lpcs_wm: 0.950 ~ 0.970`
     - `lpcs_bg: 0.279 ~ 0.290`
     - `lpcs_fg: 0.351 ~ 0.364`
     - `lpcs_dm / lpcs_rdm: -0.019 ~ -0.022`
     - `lpcs_ds: 0.022 ~ 0.027`
- 当前判断: 继续，但当前更像“机制成立”而不是“结果已赢”
- 原因:
  1. 正面信号是：
     - `comp_ctx` 不是挂件，`lpcs_ctxm` 稳定非零且量级充足
     - `fg-bg` 间隔在 `Epoch 22` 已明显拉开
     - `lpcs_dm / lpcs_ds` 都在扩大，说明 correction 正在真正改变 pair score
  2. 但截至目前：
     - `ep20 = 47.6 / 60.0`
     - 仍只是与此前 `query_ctx` 线的早期结果同量级
  3. 因此现阶段还不能说：
     - `competition context` 已经强于 `query context`
     - 只能说它已经被干净接上，而且值得继续看后续验证点

### [2026-03-22 04:50] `ep30` 出现 post-warmup dip，与历史 LPCS 一致

- 当前进度:
  - 已完成 `ep30` 验证
  - 当前处于 `Epoch 32+`
- 关键验证结果:
  - `ep30 = 41.5 / 54.7`（相比 `ep20 = 47.6 / 60.0`，下降 6.1%/5.3%）
- 当前判断: 继续，post-warmup dip 是已知现象
- 原因:
  1. `LPCS warmup=20`，`epoch 21+` 才真正引入 `lpcs loss`，叠加到已有训练信号上
  2. 此时 `LR` 正处于 cosine 下降前的较高区段，加上新 loss 的初始冲击，出现临时性能下降
  3. 历史上 `exp135 query_ctx` 也经历过类似 dip 后恢复
  4. `Epoch 23` 日志显示 `lpcs_ctxm ≈ 0.546`、`lpcs_fg/bg ≈ 0.42/0.31`，说明 `comp_ctx` 仍在正常接入
  5. 下一关键判断点：
     - `ep40`：若回到 50%+ mAP，说明 dip 已恢复
     - `ep50`：若仍低于 50%，需要考虑止损
  6. LPCS 系列对照（ep30 dip 参考）:
     - `exp135 corrected LPCS ep30`：待查
     - `exp139 query_ctx ep30`：待查

### [2026-03-22 05:14] `ep40` 出现，恢复缓慢，远落后于正常训练曲线

- 当前进度:
  - 已完成 `ep40` 验证
  - 当前处于 `Epoch 43+`
- 关键验证结果:
  - `ep40 = 43.9 / 59.8`（使用 `cvk_residual` 模式）
- 训练曲线:
  - `ep10 = 36.5 / 50.0`
  - `ep20 = 47.6 / 60.0`（warmup 结束前的峰值）
  - `ep30 = 41.5 / 54.7`（post-warmup dip）
  - `ep40 = 43.9 / 59.8`（恢复中，但仍远低于 ep20 的 mAP）
- LPCS 机制观察（E34-35）:
  - `lpcs_dm ≈ 0.276`（correction direction 增长强劲）
  - `lpcs_fg ≈ 1.02`（foreground score 已突破 1.0）
  - `lpcs_bg ≈ 0.47`（background score 适中）
  - `lpcs_fg - lpcs_bg ≈ 0.55`（fg-bg 间隔大幅拉开）
  - `lpcs_ctxm ≈ 0.555`（competition context 稳定非零）
- 当前判断: 继续但不乐观
- 原因:
  1. E40 = 43.9% 远低于 exp030a E40 = 55.6%（-11.7%）
  2. 但 exp141 使用 `cvk_residual` eval mode，与 exp030a 的 `concat_scaled` 不同
  3. LPCS 训练 loss 可能干扰了主学习，导致基础模型质量下降
  4. 需要观察 E50-60 是否能恢复到 55%+ 水平
  5. 如果 E60 仍低于 50%，应考虑止损

### [2026-03-22 05:38] `ep50` 大幅恢复，post-warmup dip 正在修复

- 当前进度:
  - 已完成 `ep50` 验证
- 关键验证结果:
  - `ep50 = 51.3 / 64.1`（比 E40 的 43.9% 大幅恢复 +7.4%!）
- 训练曲线:
  - `ep20 = 47.6 / 60.0`
  - `ep30 = 41.5 / 54.7`（post-warmup dip）
  - `ep40 = 43.9 / 59.8`
  - `ep50 = 51.3 / 64.1`（大幅恢复，已超过 E20 的 mAP）
- 当前判断: 继续，恢复趋势明确
- 原因:
  1. E50 mAP 51.3% 已显著超过 E20 的 47.6%，说明 LPCS 不只是恢复，而是真的在学
  2. 但仍远低于 exp030a E50 = 55.7%（concat_scaled mode），差距 ~4.4%
  3. exp141 使用 `cvk_residual` eval mode，与 exp030a 不同，无法直接比较
  4. 下一判断点：E60 和 E70，看是否能继续缩小差距

### [2026-03-22 06:02] `ep60` 停滞，LPCS 方向信号偏负

- 当前进度:
  - 已完成 `ep60` 验证
  - 当前处于 `Epoch 61`
- 关键验证结果:
  - `ep60 = 51.4 / 64.2`（vs E50 = 51.3 / 64.1，仅 +0.1%）
- 训练曲线:

| Epoch | mAP  | R1   | 增量 |
|-------|------|------|------|
| 20    | 47.6 | 60.0 | —    |
| 30    | 41.5 | 54.7 | -6.1 |
| 40    | 43.9 | 59.8 | +2.4 |
| 50    | 51.3 | 64.1 | +7.4 |
| 60    | 51.4 | 64.2 | +0.1 |

- LPCS 机制观察（E61）:
  - `lpcs_dm ≈ 0.420`（correction direction 仍在增长）
  - `lpcs_fg ≈ 1.51`（foreground score 很高）
  - `lpcs_bg ≈ 0.58`（background 也在增长）
  - `lpcs_fg - lpcs_bg ≈ 0.93`（间隔很大，说明 scorer 能区分）
  - 但 **mAP 没有跟着涨**，说明 scorer 虽然能区分 fg/bg，但纠正后的排序并没有变好
- 当前判断: **停滞信号明确**，但尚未达到止损条件
- 原因:
  1. E50→E60 仅 +0.1%，恢复趋势已停
  2. 远低于 exp030a E60 = 57.7%（-6.3%）
  3. LPCS scorer 在学但 mAP 不涨 → 说明 correction 方向可能系统性偏误
  4. 但 CLAUDE.md 止损条件是"连续 20 epoch 无提升"，目前只有 10 epoch（E50→E60）
  5. 等到 E70-80，如果仍停滞，果断止损
- 下一判断点: E70（若 mAP < 52%，准备止损）

### [2026-03-22 06:26] `ep70` 恢复继续，未触及止损线

- 当前进度:
  - 已完成 `ep70` 验证
- 关键验证结果:
  - `ep70 = 53.7 / 66.0`（比 E60 的 51.4% 提升 +2.3%）
- 训练曲线:

| Epoch | mAP  | R1   | 增量 |
|-------|------|------|------|
| 50    | 51.3 | 64.1 | +7.4 |
| 60    | 51.4 | 64.2 | +0.1 |
| 70    | 53.7 | 66.0 | +2.3 |

- 当前判断: 继续，但仍远低于 exp030a 同期
- 原因:
  1. E70 = 53.7% 已超过 E60 的 51.4% (+2.3%)，说明模型还在学
  2. exp030a E70 (concat_scaled) = 58.1%，差距仍有 -4.4%
  3. LPCS 方向的上界可能较低，但在跑完前无法确认
  4. 远程机器不需要空闲，继续让 exp141 跑完到 E120
- 下一判断点: E80, E90（如果 mAP > 56%，说明 LPCS 有希望；如果 < 54%，说明已近天花板）
