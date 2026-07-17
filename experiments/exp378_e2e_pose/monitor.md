# exp378 TAPF 可靠性有界任务自适应姿态场运行监控

## 当前状态

- 阶段：Gate A 必要归因；严格 hard-freeze 与显式 zero-objective-gradient SGD relaxation
  的 `residual OFF/ON` 配对对照；
- 当前 relaxation exact execution commit：`ca62c475b43f17564bb09ede90de6eed53dd2d88`；
- execution bundle SHA256：`f58be2f612e2488c53ad0187e47adf2dc30f495e3b9b7b892cbf7b3d3e2767dc`；
- 训练：同机 B0、corrected hard F0/P0、fresh MR-F0/MR-P0 均已完整结束；4090当前唯一训练为
  fresh同机D0，output=`log/occluded_duke/exp378_d0_continued_pose_s1234`、main PID=`442976`；
  3090已完成的持续pose D0只保留跨机趋势；
- 裁决：`CONTINUE_CORE_TAPF_ATTRIBUTION`。hard/relax × residual OFF/ON `2×2` 已闭合：
  residual-OFF 内生姿态场配置相对 B0 分别为 `+0.8/+0.9 mAP`，显式 relaxation 的 matched
  描述性差值仅 `+0.1 mAP`，当前 geometry residual 在 hard/relax 下均为 `-0.3 mAP`。这只停止
  当前 geometry residual 小变体，不停止 TAPF；下一阶段先补同机 R0、D0与低成本语义审计；
- 当前主实现：Stage-2 LiteHR anchor + 冻结 anchor-head 参数 + 可靠性门控 `17×4` 低带宽几何 residual
  + Gaussian 正式重渲染；
- anchor 路线：Gate A 保持约 `8.6 万` 参数的最小 LiteHR-style head；若且仅若 bootstrap
  质量门禁失败，才启用预注册 `H0` 轻量 Lite-HRNet decoder 单变量对照；
- 关键近邻：PABR 已覆盖 pose 初始化后 ReID-only 自由微调，PAFormer 覆盖训练 pose 监督与
  测试期无外部 pose 输入；后期单任务区别于 PAFormer，但不能绕开 PABR，因此朴素内部 pose
  predictor 或 HRNet head 均不得单独包装成创新。

## 启动前必须完成

- [x] 文献查新落盘并明确 PABR/PAFormer 边界；
- [x] 科学红队审查通过；
- [x] 实现/集成审查通过（最终 Gate 0 PASS）；
- [x] 默认配置 OFF，production B0/P0 shared state与初始 descriptor/featmaps exact；
- [x] 首轮所有 arm 普通 Random Erasing 统一关闭，规避 RGB/teacher 不同步；
- [x] teacher/anchor/adapted raw 域与 PSG resize→单次 sigmoid逐元素审计；
- [x] CPU shape、schedule、低带宽 Gaussian、可靠性释放与独立 2×2 梯度单测（7/7 PASS）；
- [x] 4090 P0 e1及 F0/D0/P0/J0 e11 batch64 AMP前后向、GradScaler与 optimizer delta；
- [x] 3090 P0 e11 batch64 AMP跨机复核；
- [x] epoch 1–10 pose loss不进 backbone，ReID不进 anchor/adapter；
- [x] epoch 11+ P0 teacher/pose loss精确关闭、anchor无梯度、geometry output/upstream均接收有限非零 ReID梯度；
- [ ] 运行期记录 anchor函数随 ReID主干变化的 e10→e120漂移，不声称场绝对冻结；
- [x] eval predicted-only对 correct/shuffle/None/不可索引 pose exact parity；
- [x] F0/D0/P0/J0除 mode与 OUTPUT_DIR外逐项一致，完整 TAPF init逐键 exact；
- [x] production P0 strict reload后 descriptor/featmaps exact；
- [x] 干净 exact execution commit。

## Gate 0 证据（2026-07-16）

- 4090 full-model invariants：`TAPF_MODEL_INVARIANTS_PASS`；
- 4090 batch64 AMP：P0 e1、F0/D0/P0/J0 e11全部 `TAPF_CUDA_PREFLIGHT_PASS`；
- 3090 batch64 AMP：P0 e11 `TAPF_CUDA_PREFLIGHT_PASS`；
- P0 e11两机均为 pose=None、anchor grad/delta=0，adapter output与 upstream ReID梯度同时非零；
- F0 e11 anchor/adapter均不更新；D0 e11仅 anchor更新；J0 e11 pose→anchor与 ReID→adapter独立成立；
- geometry adapter使用 FP32数值岛，解决首个约 `4.6e-8` 更新在 FP16中量化为0的问题；Swin/PSG保持 AMP；
- 原始日志与 SHA：`remote_artifacts/exp378_gate0_20260716/{4090,3090}/`，本地七份哈希全部核对 PASS；
- 最终实现审查：`experiments/exp378_e2e_pose/codex_implementation_review.md`。

Gate 0只证明实现与执行门禁成立，不代表性能结论。exact execution commit/archive 已冻结，
Gate A 已按预注册启动；运行中不得改机制或跳过控制臂。

## Gate A 启动记录（2026-07-16）

- 4090 P0：目录 `/home/afr/SOLIDER-REID-exp378-5de3b30`，输出
  `log/occluded_duke/exp378_p0_reid_only_geometry_s1234`，启动 main PID `212182`；
- 3090 D0：目录 `/root/work/SOLIDER-REID-exp378-5de3b30`，输出
  `log/occluded_duke/exp378_d0_continued_pose_s1234`，启动 main PID `1809185`；
- 两机均核对 exact HEAD、独立输出目录、唯一 main、8 个 DataLoader worker 与 GPU 占用；
- 4090 P0 已完成 e1 并进入 e6，bootstrap pose loss 从 e1 约 `2.91` 降至 e6 约 `1.40`，
  `student_fraction=0.20` 与预注册 handoff 一致；
- 3090 D0 首批 e1 日志正常，`student_fraction=0`、几何 residual 为零，符合 D0 定义；
- 两机暂无 NaN/Inf/Traceback/RuntimeError/OOM；继续运行，`<e60` 不作负裁决；
- 用户补充的轻量 HRNet 端到端设想按设计中的 `H0` 保留：只有 e10 bootstrap 质量证据明确指出
  最小 anchor 容量不足时才替换 anchor；它不改变后期 ReID-only TAPF 机制，也不单独作为创新 claim。

## Gate A 运行记录

### 2026-07-16：4090 P0 e10 handoff / bootstrap 门禁

- e10 `student_fraction=1.0`，e11 起 teacher、pose loss 与 teacher 统计精确归零；
- bootstrap shape loss 从 e1 约 `2.75` 降至 e10 `0.851`，confidence loss 从约 `0.66`
  降至 `0.459`；anchor confidence 为 `0.763`，teacher confidence 为 `0.838`；
- anchor sigma mean 从初始上限附近 `0.250` 收缩到 `0.151`，没有出现 NaN、坍缩为全零或数值域异常；
- e10 predicted-only 完整 eval 为 `36.5 mAP / 45.0 R1 / 61.9 R5 / 68.2 R10`；
- 结论：最小 LiteHR-style anchor 已明确学习且 handoff 正常，当前不触发 `H0`。该结论只回答
  bootstrap 可用性，不是性能 GO；继续到至少 e60。

### 2026-07-16 08:00：常规健康检查

- 4090 P0 已完成 e38并进入 e39；唯一 main `212182`、8 workers、约 `7.2 GiB`，exact HEAD不变；
- 3090 D0 已完成 e17；唯一 main `1809185`、8 workers、约 `8.2 GiB`，exact HEAD不变；
- 两臂均无 NaN/Inf/Traceback/RuntimeError/OOM，teacher/student课程与各自 arm 定义一致；
- P0 日志以三位小数显示 `shift_rms/log_scale_rms=0.000`，但 e10→e20→e30→e40 checkpoint
  中零初始化 adapter 输出层权重 norm 为 `0 → 4.98e-4 → 6.92e-4 → 7.98e-4`，bias norm为
  `0 → 7.03e-4 → 9.89e-4 → 1.12e-3`，证明 FP32 adapter 正在持续接收 ReID 更新，并非死路；
- 现阶段几何改变量仍很小，继续观察到 e60并以同机 F0/D0等对照判断实际效应，不提前调尺度。

### 2026-07-16 08:15：4090 P0 e60 首轮趋势门禁

- P0 e60 完整 eval 为 `55.7 mAP / 67.3 R1 / 79.3 R5 / 83.5 R10`，e10→e60 mAP
  轨迹为 `36.5→46.2→51.8→53.4→54.0→55.7`，仍在上升；
- 历史同一4090 clean B0 e60为 `55.2/65.0`，但该历史 arm 使用 `RE_PROB=0.5`，当前 exp378
  全臂预注册为 `RE_PROB=0.0`，因此这里只能作弱参考，不能冒充 exact 同配方 B0；
- P0 没有触发“相对参考 B0 低至少 `0.5 mAP`”的快速止损条件，且数值/进程均健康，裁决为
  `CONTINUE_WITH_FUEL`，不是正式 GO；P0 完成释放4090后必须首先补 exact exp378 B0；
- 3090 D0 e20为 `46.1/56.9/71.1/76.7`，与4090 P0 e20 `46.2/56.8/71.2/76.2`
  接近，但跨运行时差值不作归因；D0继续运行；
- `H0` 仍不触发：目前没有 anchor bootstrap 容量不足证据。

### 2026-07-16 08:30：常规健康检查

- 4090本地转发一度消失，已只恢复 SSH 转发；远端 P0 main `212182` 与8 workers从未中断，
  exact HEAD、checkpoint与训练连续性均保持；
- P0 已完成 e97并进入 e98，e70/e80/e90 mAP 为 `55.3/55.7/56.0`，继续正常收敛到 e120；
- 3090 D0 已完成 e40，e30为 `52.5/64.3/76.7/81.4`；与 P0 e30的跨机差值不作归因；
- 两臂 GPU、teacher课程、anchor统计与数值域正常，全文无 NaN/Inf/Traceback/RuntimeError/OOM；
  本轮无新裁决。

### 2026-07-16 08:45：P0 完成，启动 exact B0

- 4090 P0 完整正常结束，e120=`56.2 mAP / 67.8 R1 / 79.6 R5 / 83.7 R10`；main与workers
  自然退出，GPU释放，日志结束于完整四项且无 NaN/Inf/Traceback/RuntimeError/OOM；
- P0 e100/e110/e120 mAP为 `56.0/56.1/56.2`，后段已基本收敛；
- P0 相对历史同机 B0 final `58.4/67.1` 为 `-2.2 mAP/+0.7 R1`，但历史 B0 的
  `RE_PROB=0.5` 与当前全臂 `0.0` 不同，禁止据此作正式 P0−B0裁决；
- 4090 已从同一 exact execution commit 启动预注册 `exp378_b0_clean.yml`，输出
  `log/occluded_duke/exp378_b0_clean_s1234`，main PID `245998`；唯一main、8 workers、GPU与首批
  e1日志正常；
- 3090 D0 e50=`54.4/66.1/78.4/82.5`，继续运行；跨机差值仍不作归因；
- 当前裁决保持 `PENDING_EXACT_CONTROLS`。先完成 exact B0，再按设计补 F0/D0/J0/R0与Gate B。

### 2026-07-16 09:00：exact B0 早期轨迹

- 4090 exact B0 已完成 e28并进入 e29；唯一 main `245998`、8 workers、约 `7.1 GiB`，exact HEAD
  与输出目录正确，日志无异常；
- B0 e10/e20为 `36.9/47.0/61.8/68.3` 与 `42.3/53.7/65.7/70.8`；对应 P0−B0 mAP
  暂为 `-0.4/+3.9`，但 e20仍属早期，只记录不裁决；
- 3090宿主的 Tailscale连接在本轮检查中无响应；最后确认 D0 e53仍健康、唯一main与8 workers正常。
  没有训练失败证据，禁止因监控链路中断而重启或启动替代 D0，等待链路恢复后补齐 eval；
- 当前无新性能裁决，继续 exact B0与既有 D0。

### 2026-07-16 09:15：exact B0 e50 轨迹

- 4090 B0 已完成 e56并进入 e57；唯一main、8 workers、GPU与 exact HEAD正常，日志无异常；
- B0 e30/e40/e50为 `50.6/61.9`、`53.0/65.1`、`52.1/63.5`（mAP/R1）；对应
  P0−B0 为 `+1.2/+1.4`、`+0.4/-0.3`、`+1.9/+1.9`；
- 截至 e50，除 e10外四个 exact 同机评测点的 P0 mAP均高于 B0，方向有燃料，但仍等待 e60
  与最终 e120，禁止把中途差值写成正式增益；
- 3090宿主仍未恢复可用的只读会话；继续保留 D0原进程，不作任何替代启动。

### 2026-07-16 09:30：exact B0 e80 与 D0 链路恢复

- 4090 B0 已完成 e82并进入 e83；P0−B0 在 e60/e70/e80的 mAP/R1分别为
  `+1.9/+2.1`、`+0.9/+0.2`、`+1.1/+0.5`，三个中后段 exact 同机点方向一致；
- 该轨迹已满足“继续补强对照”的燃料门槛，但正式 `P0−B0 >= +0.8 mAP`仍以 final为准；
- 3090只读链路已恢复，确认 D0原 main `1809185` 与8 workers从未中断，已完成 e90训练；
- D0 e60/e70/e80 mAP为 `55.1/55.2/55.4`，相对 P0为 `+0.6/+0.1/+0.3`。跨机趋势
  尚不足以归因给 ReID-only residual，必须继续补4090同机 D0/F0；
- 两机 exact HEAD、GPU与日志正常，无 NaN/Inf/Traceback/RuntimeError/OOM；不触发 H0。

### 2026-07-16 13:25：B0/D0 完成，进入 F0（后续冻结审计推翻 GO）

- 中间多轮 heartbeat 因本地 Tailscale TCP通道异常未能执行远端动作；已通过只读
  `tailscale nc` ProxyCommand恢复两机转发。远端训练均未受影响，禁止把监控空窗写成执行中断；
- 4090 exact B0 已于 e120完整结束：`55.1 mAP / 66.7 R1 / 79.5 R5 / 83.8 R10`；
  进程/workers自然退出，GPU释放，日志无 NaN/Inf/Traceback/RuntimeError/OOM；
- 当时观测到同机 final `P0−B0 = +1.1 mAP / +1.1 R1 / +0.1 R5 / -0.1 R10`；
  但 13:49 的 checkpoint 参数审计发现 P0 anchor 未真正冻结，因此该差值不再计入预注册门禁，
  只能保留为 bug execution 的诊断数值；B0 本身仍有效；
- 3090 D0 已完整结束：`55.7/67.6/79.0/83.2`；跨运行时 final P0−D0为
  `+0.5/+0.2/+0.6/+0.5`，刚到 mAP探索门槛但不能替代同机归因；
- 4090 已按预注册顺序启动 F0，输出 `log/occluded_duke/exp378_f0_frozen_anchor_s1234`，main
  PID `279592`；exact HEAD、唯一main、8 workers、约 `7.3 GiB`与首批 e1 TAPF日志正常；
- 该时点裁决曾为 `GO_FOR_EXACT_F0`，已被 13:49 的 `INVALID_MOMENTUM_DRIFT` 覆盖。

### 2026-07-16 13:33：F0 完成 e10 handoff（后续冻结审计判无效）

- 4090 F0 已完成 e10完整评测：`37.6 mAP / 46.7 R1 / 63.0 R5 / 69.5 R10`，随后正常进入
  e11；当前已完成 e13并运行 e14；
- bootstrap 阶段 shape loss 从 e1约 `2.75`下降至 e10约 `0.84`，anchor confidence 从约
  `0.54`升至约 `0.76`，说明冻结前的姿态锚点已正常建立；
- e11之后 `tapf_pose/shape_loss/confidence_loss/teacher_confidence/valid_fraction`均精确为零，
  teacher raw/sigmoid统计也全部为零；`shift_rms/log_scale_rms/low_conf_shift/high_conf_shift`
  持续为零，且 adapted raw 与 anchor raw逐项一致；这些日志只证明 adapter 输出关闭，不能证明
  anchor 参数冻结，后续 checkpoint 参数比较已证实 anchor 仍被 optimizer 改写；
- exact HEAD仍为 `5de3b3007b0bd9c5946af47fc79bf85ed10b2e2e`，远端只有 main
  `279592`及其8个 DataLoader workers，GPU和日志持续前进；没有
  NaN/Infinity/Traceback/RuntimeError/OOM；
- e10仍属 bootstrap 结束点，不作性能裁决；继续运行至 e120后才计算正式 `P0−F0`。

### 2026-07-16 13:49：SGD momentum 暴露硬冻结 bug，P0/F0 均转为不可报告

- 对 F0 checkpoint 做逐参数比较后发现：e10→e20与 e10→e30均有全部 `26/26` 个 anchor
  参数变化，最大绝对变化分别为 `0.001771`与 `0.003057`；geometry adapter 的 `6/6` 个参数
  则保持逐位相同；
- 反查已完成 P0 后确认同一问题：e10→e120的 anchor `26/26` 个参数变化，最大绝对变化
  `0.008602`；adapter `6/6` 参数变化是 P0 预期的 ReID写入，但不能抵消 anchor 冻结失效；
- 根因已精确定位：冻结运行时为 PyTorch `1.13.1+cu117`，其
  `optimizer.zero_grad(set_to_none=False)`是默认行为。e10 bootstrap 留下的 anchor 梯度张量被
  清零而非置为 `None`，SGD momentum与 weight decay遂在无新 objective gradient时仍继续更新
  anchor。原单步 preflight显式使用 `set_to_none=True`，没有复现生产路径的陈旧 momentum状态；
- 当前 F0 在 e45中途只对 main `279592`发送 TERM，8 workers随主进程退出，GPU释放；原目录、
  e10/e20/e30/e40 checkpoints与日志全部保留，禁止继续或发布；F0已完成的 e20/e30/e40 eval
  分别为 `47.6/58.1/71.9/77.2`、`51.9/63.3/75.5/80.5`、
  `54.3/65.7/79.2/82.9`，仅作 bug 诊断；
- 当前状态改为 `INCOMPLETE_NONREPORTABLE / INVALID_MOMENTUM_DRIFT`。B0与持续 pose监督 D0
  不依赖 post-bootstrap hard freeze，结果仍有效；原 P0与当前 F0必须在显式硬冻结修复后从头
  重跑，不能从旧 checkpoint续训；
- 最小修复限定为：P0/F0在 e11起把 anchor参数设为 `requires_grad=False`并清空残留 `.grad`；
  D0/J0保持可训练。新增回归必须复现 PyTorch 1.13 默认 zero-grad + SGD momentum/weight decay，
  并证明 e11 optimizer step后 anchor逐位不变；修复审查通过后先重跑正确 P0，再补同机 F0。

### 2026-07-16 14:00：hard-freeze 修复通过，fresh P0 已启动

- 最小修复已提交为 exact execution commit
  `f1cf1ea70cf39be95e5e8e094430909df61b0739`；传输 bundle SHA256为
  `779b511d4f5db643f302e10c267914776ad9d91b9ec65a5d15c693be078422d4`；
- 4090冻结 PyTorch 1.13运行时的七项 CPU回归全部通过，其中新增回归已真实建立 e10 SGD
  momentum/weight-decay状态，再用生产默认 `zero_grad()`验证 P0/F0 e11后 anchor逐位不变；
- batch64 CUDA preflight通过：P0 e11的 anchor grad/delta均为 `0`，adapter grad
  `1.0809e-3`、delta `3.7460e-6`；F0 e11的 anchor与adapter grad/delta均为 `0`；两者
  teacher/pose loss均关闭，标准 ReID梯度仍到达 backbone；
- 修复审查结论为 `PASS_FOR_FRESH_EXECUTION`。4090 已从新 exact commit从头启动修复后 P0，
  output=`log/occluded_duke/exp378_p0_reid_only_geometry_hardfreeze_s1234`，main PID=`296008`；
  启动时唯一main与8个 DataLoader workers、GPU、首个 forward均正常；
- 旧 P0与旧 F0继续保留为 `INVALID_MOMENTUM_DRIFT`，不恢复、不续训。当前所有 TAPF正式门禁
  重新置为 pending；修复后 P0完成前不得继续引用旧 `+1.1 mAP`为方法增益。

### 2026-07-16 14:02：fresh P0 bootstrap 健康

- exact HEAD仍为 `f1cf1ea70cf39be95e5e8e094430909df61b0739`；唯一 main `296008`
  与8个 DataLoader workers持续运行，GPU约 `7.3 GiB`、利用率约 `88%`；
- 已完整结束 e6并运行 e7；e6→e7的 student fraction按课程从 `0.2`升至 `0.4`，shape loss
  约从 `1.24`继续降至 `1.06`，anchor confidence约 `0.66→0.69`；bootstrap阶段 residual
  shift/log-scale保持为零，符合预注册日程；
- runner/train log持续增长，无 NaN/Infinity/Traceback/RuntimeError/OOM；尚无新完整 eval，继续
  等待 e10 handoff与 e20 checkpoint硬冻结参数审计。

### 2026-07-16 14:18：fresh P0 hard-freeze checkpoint 门禁通过

- fresh P0 已完成 e10/e20/e30完整评测，依次为 `37.2/46.6/62.5/68.9`、
  `46.5/57.1/70.2/76.2`、`51.6/63.3/74.5/79.7`；当前已完成 e36并运行 e37；
- 对 e10/e20/e30 checkpoint逐参数比较：e10→e20和 e10→e30的 anchor均为
  `0/26 changed`，最大绝对差与整体 L2差均精确为 `0`，证明生产训练中的 hard freeze真正生效；
- 同期 geometry adapter均为 `6/6 changed`：e10→e20最大绝对差 `0.001219`、L2差
  `0.02459`，e10→e30最大绝对差 `0.002746`、L2差 `0.05531`。这证明 P0在冻结 anchor后
  仍有有限的 ReID写入，并非死模块；日志三位小数显示的 shift/log-scale `0.000`只是精度截断，
  不能解释为参数未更新；
- e11后 teacher raw/sigmoid、pose/shape/confidence loss持续精确为零；exact HEAD、唯一main与8
  workers、GPU和日志均健康，无 NaN/Infinity/Traceback/RuntimeError/OOM；
- `INVALID_MOMENTUM_DRIFT`已被修复后 checkpoint证据排除。当前裁决为
  `CONTINUE_CORRECTED_P0`；e30仍早于性能门禁，不与 B0作负裁决。

### 2026-07-16 14:33：fresh P0 e60 保持正向燃料

- fresh P0 e40/e50/e60完整评测依次为 `52.7/63.9/76.0/80.6`、
  `53.7/65.1/76.8/81.3`、`54.6/65.8/78.1/82.4`；当前已完成 e65并继续运行；
- 有效同机 B0 e60为 `53.8/65.2/77.7/82.0`，因此 corrected P0−B0在 e60暂为
  `+0.8 mAP / +0.6 R1 / +0.4 R5 / +0.4 R10`；方向达到继续燃料，但正式
  `P0−B0 >= +0.8 mAP`仍只以 final e120裁决；
- e10→e60 checkpoint参数审计继续通过：anchor `0/26 changed`、最大绝对差与 L2差均精确为
  `0`；adapter `6/6 changed`、最大绝对差 `0.006239`、L2差 `0.12569`；
- e11后 teacher与 pose loss持续关闭，exact HEAD、唯一main与8 workers、GPU和日志均健康，无
  NaN/Infinity/Traceback/RuntimeError/OOM；当前裁决保持 `CONTINUE_CORRECTED_P0`。

### 2026-07-16 14:48：fresh P0 e70–e90 中后段轨迹

- fresh P0 e70/e80/e90完整评测为 `55.0/66.1/78.4/82.9`、
  `54.9/65.9/77.6/82.5`、`55.7/67.4/78.8/83.4`；当前已完成 e94并运行 e95；
- 相对有效同机 B0，对应 mAP差为 `+0.6/+0.3/+0.8`；R1差为 `-0.3/-0.7/+1.0`。
  mAP方向在三个点均为正，但幅度波动且 R1并非全程占优，继续等待 final，不提前触发 F0；
- e10→e90参数审计仍为 anchor `0/26 changed`、最大绝对差与 L2差精确 `0`；adapter
  `6/6 changed`、最大绝对差 `0.007855`、L2差 `0.15823`；
- teacher/pose loss保持关闭，唯一main与8 workers、GPU和日志正常，无
  NaN/Infinity/Traceback/RuntimeError/OOM；裁决保持 `CONTINUE_CORRECTED_P0`。

### 2026-07-16 15:03：fresh P0 final 未过原门槛，继续必要归因

- fresh corrected P0 已完整自然结束；e100/e110/e120为 `55.5/66.7/78.2/83.1`、
  `55.5/66.7/78.3/83.0`、`55.6/66.7/78.4/83.0`，main与 workers自然退出，GPU释放；
- 相对有效同机 B0 final `55.1/66.7/79.5/83.8`，corrected P0−B0为
  `+0.5 mAP / +0.0 R1 / -1.1 R5 / -0.8 R10`。mAP未达到预注册 `+0.8`，且 R1无增益、
  高 rank退化；该门槛事实保留，但用户明确要求不要据此提前终止 TAPF探索；
- final参数审计通过：e10→e120 anchor `0/26 changed`，最大绝对差与 L2差精确为 `0`；
  adapter `6/6 changed`，最大绝对差 `0.008120`、L2差 `0.16354`。因此此次原门槛未通过不是
  hard-freeze再次失效或 adapter死亡，而是修复后机制本身未达到首轮性能门槛；
- 全程 e11后 teacher/pose loss关闭，无 NaN/Infinity/Traceback/RuntimeError/OOM。旧 P0/F0仍为
  对 hard-freeze claim无效，但其 `+1.1 mAP`不再丢弃：重新保留为“隐式 momentum-relaxed
  anchor”先导信号，后续需显式复现，而不能继续误称 hard freeze；
- 当前裁决改为 `CONTINUE_ATTRIBUTION_OVERRIDE`。首先补同机 corrected F0，直接判断严格冻结条件下
  `+0.5 mAP`是否来自 ReID几何 residual；随后显式实现并配对验证 momentum-relaxed P0/F0，
  再决定 D0/J0/R0与 Gate B的最小必要集合；
- H0目前仍不触发：bootstrap质量正常，轻量 Lite-HRNet容量不是当前首要未决变量。更强 anchor
  只能在现有 hard/soft-freeze与 residual归因完成后按独立设计评估。

### 2026-07-16 15:21：fresh corrected hard-freeze F0 已启动

- 4090转发消失后仅通过 `tailscale nc` 恢复 SSH 端口转发；远端已完成 P0及既有资产未改动；
- exact HEAD核对为 `f1cf1ea70cf39be95e5e8e094430909df61b0739`，启动前无训练 main，GPU空闲；
- fresh F0输出为 `log/occluded_duke/exp378_f0_frozen_anchor_hardfreeze_s1234`，唯一 main PID
  `330895`，8个子进程均为 DataLoader workers；启动后 GPU约 `7.3 GiB`、利用率约 `89%`；
- e1日志已持续前进，`POSE_TAPF_MODE=f0`，bootstrap pose loss约 `2.92`，adapter shift/log-scale
  均为零；teacher/raw/sigmoid统计、batch64、AMP日程正常，无 NaN/Inf/Traceback/RuntimeError/OOM；
- e20 checkpoint落盘后必须比较 e10→e20：anchor要求 `0/26 changed`，adapter要求
  `0/6 changed`。完成后计算 corrected hard `P0-F0`，随后无论正负都继续 matched explicit
  relaxation F0/P0，不据此直接终止 TAPF。

### 2026-07-16 15:28：corrected F0 e10 handoff 健康

- exact HEAD仍为 `f1cf1ea70cf39be95e5e8e094430909df61b0739`；唯一 main `330895`
  与8个 DataLoader workers正常，GPU约 `7.1 GiB`、利用率约 `93%`；
- e10完整 eval=`37.8 mAP / 47.6 R1 / 62.9 R5 / 69.4 R10`，为 bootstrap/handoff
  端点，不作性能裁决；
- e11后 teacher raw/sigmoid、pose/shape/confidence loss均精确为零，student fraction为 `1.0`；
  F0的 shift/log-scale持续为零，adapted field与anchor一致，符合 residual OFF定义；
- 已进入 e14，日志无 NaN/Infinity/Traceback/RuntimeError/OOM。继续到 e20 checkpoint后执行
  anchor `0/26`、adapter `0/6` 的逐参数 hard-freeze审计。

### 2026-07-16 15:36：corrected F0 e20 hard-freeze门禁通过

- e20完整 eval=`46.3 mAP / 57.3 R1 / 72.1 R5 / 76.8 R10`；当前已进入 e30，
  `<e60`仅记录轨迹，不作性能负裁决；
- e10→e20逐参数比较通过：anchor `0/26 changed`、最大绝对差 `0`、L2差 `0`；adapter
  `0/6 changed`、最大绝对差 `0`、L2差 `0`；
- 这同时证明 production hard freeze生效且 F0 residual确实关闭，不存在旧执行的 SGD
  momentum/weight-decay漂移；teacher与pose loss继续为零，日志无数值或运行时异常；
- F0继续到 e120。显式 relaxation 已另立设计并在本地实现/审查，禁止在当前运行中修改远端代码。

### 2026-07-16 15:43：corrected F0 e30–e40 健康

- e30/e40完整 eval分别为 `52.2/63.5/75.7/80.5` 与 `53.0/64.2/76.6/81.6`；
- 唯一 main `330895`、8 workers与 GPU约 `7.1 GiB`均正常，teacher/pose loss及 residual
  继续精确关闭，日志无 NaN/Infinity/Traceback/RuntimeError/OOM；
- 当前仍早于 e60，只记录曲线并继续；不把 corrected P0−F0中途差值用于归因或停止决策。

### 2026-07-16 15:54：显式 relaxation 候选通过3090兼容 preflight

- 显式 zero-objective-gradient SGD relaxation已完成独立 design、两轮 Codex阻塞性审查与实现，
  candidate commit=`8af76a10409b666279ffc2f330fc5422b4902cd6`；传输bundle SHA256为
  `179378e387fa7fdef5dbab110f889fb69843011ccbfee6b617aa2f3f4f415642`；
- 3090实际 runtime为 PyTorch `2.4.1+cu121`，故本轮只作跨运行时兼容证据，不能替代4090
  PyTorch `1.13.1+cu117`最终放行；full-model invariants与12项CPU单元均通过；
- MR-F0 batch64 e11 CUDA preflight通过：10步 legacy parity、真实overflow `128→64`，anchor
  gradient `0`、delta `9.9231e-4`，adapter gradient/delta均为 `0`；
- MR-P0 batch64 e11 CUDA preflight通过：10步 legacy parity、真实overflow `128→64`，anchor
  gradient `0`、delta `8.9309e-4`，adapter gradient `1.0926e-3`、delta `3.7493e-6`；
- 两臂pose loss均为 `None`，外部pose不被读取，eval pose-input exact parity通过；原始日志已回传
  `remote_artifacts/exp378_relax_preflight_8af76a1/3090/`，SHA分别为
  `1db3f48a...74221a`与`76cfa410...d068b`；
- 4090当前 corrected F0不受影响并继续运行。其完成后必须先在原PyTorch1.13 runtime重复
  invariants与两份CUDA preflight，全部通过后才可串行启动fresh MR-F0，禁止直接使用3090 PASS放行。

### 2026-07-16 15:56：corrected F0 e50–e60，hard residual暂呈负向

- F0 e50/e60完整 eval分别为 `54.1/65.5/78.4/83.0` 与 `55.2/67.3/78.9/82.9`；
- corrected hard P0 e50/e60为 `53.7/65.1/76.8/81.3` 与 `54.6/65.8/78.1/82.4`，
  因而同机 `P0-F0` 暂为 e50 `-0.4/-0.4/-1.6/-1.7`、e60
  `-0.6/-1.5/-0.8/-0.5`；
- 该趋势说明严格 hard-freeze条件下 geometry residual当前没有显示独立收益，但不是全线停止条件；
  F0已进入 e68并继续到 final，随后仍按计划完成显式 relaxation MR-F0/MR-P0；
- teacher/pose loss与 residual保持关闭，进程/GPU/日志正常，无 NaN/Inf或运行时异常。

### 2026-07-16 15:58：corrected F0 e70 健康

- F0 e70完整 eval=`55.6/67.6/79.6/84.0`；相对 corrected hard P0 e70
  `55.0/66.1/78.4/82.9`，`P0-F0=-0.6/-1.5/-1.2/-1.1`；
- hard residual负向趋势与 e60一致，但仍只属于单一 transition条件下的归因结果，不触发
  TAPF全线停止；继续等待final并执行matched relaxation 2×2；
- exact HEAD、唯一main/8 workers、GPU及日志均健康，teacher/pose loss和 residual保持关闭，
  无 NaN/Infinity/Traceback/RuntimeError/OOM。

### 2026-07-16 16:14：corrected F0 e80–e100 健康

- F0 e80/e90/e100完整 eval依次为 `55.5/66.4/79.2/83.0`、
  `55.8/67.2/79.5/83.3`、`55.7/67.1/79.2/83.2`；相对 corrected hard P0同 epoch，
  `P0-F0` mAP依次为 `-0.6/-0.1/-0.2`，严格冻结条件下 residual仍未显示稳定独立收益；
- exact HEAD仍为 `f1cf1ea70cf39be95e5e8e094430909df61b0739`；唯一 main `330895`
  与8个 DataLoader workers正常，已完成 e103并运行 e104，GPU约 `7.1 GiB`、利用率约 `91%`；
- e11后 teacher、pose loss与 residual继续精确关闭，anchor/adapted统计一致，日志无
  NaN/Infinity/Traceback/RuntimeError/OOM；继续等待 e110/e120和 final参数审计，不以当前趋势
  提前停止 TAPF，随后仍执行 matched explicit relaxation 2×2。

### 2026-07-16 16:29：corrected hard 2×2 第一行完成，姿态锚点本身达到 +0.8 mAP

- corrected F0 已完整自然结束，e110/e120均为 `55.9 mAP`，对应完整指标分别为
  `55.9/67.5/79.4/83.4`与 final `55.9/67.4/79.3/83.3`；main与8 workers自然退出，
  GPU释放，全程无 NaN/Infinity/Traceback/RuntimeError/OOM；
- 相对有效同机 B0 final `55.1/66.7/79.5/83.8`，F0−B0为
  `+0.8 mAP / +0.7 R1 / -0.2 R5 / -0.5 R10`。这说明 bootstrap 后严格冻结的内生姿态锚点
  本身已有明确 mAP/R1燃料，TAPF并未相对 baseline变负；
- corrected hard P0−F0 final为 `-0.3/-0.7/-0.9/-0.3`。因此 hard条件下的当前 geometry residual
  没有独立贡献，反而轻度削弱姿态锚点；该结论只归因 residual，不终止 TAPF或“不冻结”路线；
- e10→e120逐参数审计通过：anchor `0/26 changed`、adapter `0/6 changed`，两组最大绝对差与
  L2差均精确为 `0`且全部有限。hard-freeze × residual OFF/ON这一行至此证据闭合；下一步按既定
  方案在4090 PyTorch 1.13运行时验证并运行 matched explicit relaxation MR-F0/MR-P0。

### 2026-07-16 17:04：4090原生 relaxation 门禁全通过，放行 fresh MR-F0

- 新 exact execution commit为 `ca62c475b43f17564bb09ede90de6eed53dd2d88`，完整 bundle
  SHA256=`f58be2f612e2488c53ad0187e47adf2dc30f495e3b9b7b892cbf7b3d3e2767dc`；
  4090独立repo=`/home/afr/SOLIDER-REID-exp378-9620a20`，runtime已核对为
  PyTorch `1.13.1+cu117` / CUDA `11.7`；
- 补入生产 AMP可观测性：周期日志记录主 optimizer的scale、采样step skip及fresh process累计
  skip；每次真实overflow即时warning。两路独立Codex复审均为 `PASS_NO_BLOCKER`，新增纯函数
  状态机preflight要求 normal/overflow/recovery=`0/1/0`且累计为`1`；
- full-model paired initialization与strict reload通过；生产运行时12项单元为 `12/12 PASS`。
  原测试在 PyTorch 1.13对测试内自建 optimizer BytesIO使用受限反序列化失败，已仅在测试中显式
  改为可信 `weights_only=False`并重新审查，不触及训练路径；
- 4090 MR-F0 e11 batch64 preflight通过：`runtime_parity_steps=10`、真实overflow
  `128→64`、anchor objective gradient=`0`、anchor delta=`9.92050212e-4`、adapter delta=`0`；
- 4090 MR-P0 e11 batch64 preflight通过：同样10步parity与overflow，anchor objective gradient=`0`、
  anchor delta=`8.92860208e-4`、adapter gradient=`1.08199176e-3`、adapter delta=`3.74626070e-6`；
- 两臂pose loss均为 `None`，外部pose不读取，eval pose-input exact parity通过。原始日志已回传
  `remote_artifacts/exp378_relax_preflight_ca62c47/4090/`并逐文件SHA核对。所有生产门禁已满足，
  下一步只串行启动fresh MR-F0；MR-P0必须等待MR-F0完整结束，禁止并行或续训。

### 2026-07-16 17:05：fresh MR-F0 已启动，AMP事件日志实机生效

- 4090 fresh MR-F0已从 exact commit `ca62c475b43f17564bb09ede90de6eed53dd2d88`
  启动，output=`log/occluded_duke/exp378_mrf0_sgd_relax_s1234`，唯一 main PID=`372350`；
  8个子进程均为 DataLoader workers，GPU约 `7.3 GiB`、利用率约 `87%`；
- production AMP可观测性已在真实训练触发：e1 iter4检测到一次 main-optimizer overflow，scale
  `1024→512`并即时warning，随后周期日志均显示 `tapf_amp_skip=0`、
  `tapf_amp_skip_total=1`，证明未漏记采样间隔内事件；本次overflow由GradScaler安全整步跳过，
  不重启、不手工干预；
- e1完整结束并进入e2；pose bootstrap正常，pose loss约从 `2.92`下降，anchor confidence约
  `0.54→0.55`，MR-F0的geometry residual保持为零。日志无
  NaN/Infinity/Traceback/RuntimeError/OOM；继续到e10/e20后执行anchor relaxation轨迹、adapter
  `0/6 changed`及AMP事件序列审计，MR-P0仍禁止提前启动。

### 2026-07-16 17:05：MR-F0 e1–e8 bootstrap轨迹健康

- exact HEAD仍为 `ca62c475b43f17564bb09ede90de6eed53dd2d88`；唯一 main `372350`
  与8个 DataLoader workers正常，GPU约 `7.3 GiB`、利用率约 `87%`；已完成e7并运行e8；
- curriculum按预注册从e6/e7/e8的 student fraction `0.2/0.4/0.6`推进，pose loss从e1约`2.92`
  降至e8约`1.07`，anchor confidence升至约`0.72`；MR-F0 residual持续为零；
- AMP scale保持`512`，除e1 iter4外无新增skip，fresh-process累计仍为`1`。无
  NaN/Infinity/Traceback/RuntimeError/OOM；尚未到e10完整eval与handoff，只记录健康轨迹并继续。

### 2026-07-16 17:20：MR-F0 e10→e30 relaxation门禁通过

- MR-F0 e10/e20/e30完整eval依次为 `37.9/47.1/63.2/69.5`、
  `46.5/57.6/71.7/76.5`、`51.6/62.9/75.9/80.2`；当前已完成e36并运行e37，
  `<e60`只记录轨迹，不作性能负裁决；
- 相对同epoch直接单变量对照 corrected hard F0，MR-F0−hard F0在e10/e20/e30依次为
  `+0.1/-0.5/+0.3/+0.1`、`+0.2/+0.3/-0.4/-0.3`、
  `-0.6/-0.6/+0.2/-0.3`（顺序均为mAP/R1/R5/R10）；当前波动混合，尚不能提前判断
  relaxation主效应；
- e11后 teacher raw/sigmoid与pose/shape/confidence loss均精确为零，`anchor_relax_active=1`；
  relaxation momentum norm从e11约`0.038`衰减至约`0.025`，anchor parameter norm由约
  `24.737`平滑降至约`24.648`，符合零新目标梯度的SGD状态松弛；
- checkpoint逐参数审计通过：e10→e20 anchor `26/26 changed`、最大绝对差`0.002071`、
  L2差`0.035173`；e10→e30 anchor `26/26 changed`、最大绝对差`0.002955`、L2差`0.070010`；
  两个时间点adapter均为`0/6 changed`、最大差与L2差精确为零，且所有差值有限；
- AMP scale已按正常增长升至`4096`，无新增overflow，fresh-process skip累计仍为`1`；exact HEAD、
  唯一main/8 workers、GPU及日志健康，无NaN/Infinity/Traceback/RuntimeError/OOM。继续到final，
  MR-P0仍禁止提前启动。

### 2026-07-16 17:38：MR-F0 e40–e70 eval与matched差值

- exact HEAD仍为 `ca62c475b43f17564bb09ede90de6eed53dd2d88`；唯一main `372350`
  与8个DataLoader workers正常，已完整结束e70并继续运行；GPU约`7.1 GiB`、利用率约`93%`；
- MR-F0 e40/e50/e60/e70完整eval依次为 `52.9/64.9/77.7/81.8`、
  `53.6/64.4/78.0/82.4`、`55.2/66.9/78.9/83.3`、`55.2/66.7/78.7/82.9`；
- 相对同epoch直接单变量对照 corrected hard F0，MR-F0−hard F0依次为
  `-0.1/+0.7/+1.1/+0.2`、`-0.5/-1.1/-0.4/-0.6`、
  `+0.0/-0.4/+0.0/+0.4`、`-0.4/-0.9/-0.9/-1.1`（顺序均为mAP/R1/R5/R10）；
  当前尚未形成稳定正增益，但按预注册继续完整运行，不因单一epoch或单一门槛提前停止；
- e11后teacher与pose相关loss继续精确为零，`anchor_relax_active=1`，residual/adapter统计保持零；
  relaxation momentum norm约`0.025`，anchor parameter norm平滑降至约`24.565`；AMP scale已增长到
  `65536`，除e1安全overflow外无新增skip，fresh-process累计仍为`1`。无
  NaN/Infinity/Traceback/RuntimeError/OOM；MR-P0仍禁止提前启动。

### 2026-07-16 17:53：MR-F0 e80–e100 matched轨迹

- exact HEAD仍为 `ca62c475b43f17564bb09ede90de6eed53dd2d88`；唯一main `372350`
  与8个DataLoader workers正常，已完整结束e100并继续运行；GPU约`7.1 GiB`、利用率约`87%`；
- MR-F0 e80/e90/e100完整eval分别为 `55.2/65.8/78.7/82.7`、`56.0/67.3/79.3/83.7`、
  `55.9/66.8/79.1/83.3`；相对同epoch corrected hard F0分别为
  `-0.3/-0.6/-0.5/-0.3`、`+0.2/+0.1/-0.2/+0.4`、
  `+0.2/-0.3/-0.1/+0.1`（顺序为mAP/R1/R5/R10）。e90/e100出现轻微正向混合差值，
  仍只作为单seed轨迹，继续等待e110/final；
- e11后的teacher/pose loss、MR-F0 adapter更新仍为零，anchor relaxation momentum norm约
  `0.025`，parameter norm由e70约`24.565`平滑降至e99约`24.539`；
- AMP scale在e76/e87/e97增长到`131072`后，分别在e78/e88/e98安全回退到`65536`；
  fresh-process skip累计为`4`。每次回退后训练均继续、loss有限，未见
  NaN/Infinity/Traceback/RuntimeError/OOM，因此不重启、不改运行中配置。

### 2026-07-16 18:09：MR-F0 final有效，fresh MR-P0已串行启动

- MR-F0已完整结束且main/workers全部退出，e110/final分别为 `56.0/67.1/79.1/83.3`、
  `56.0/67.1/79.2/83.4`；相对同epoch corrected hard F0分别为
  `+0.1/-0.4/-0.3/-0.1`、`+0.1/-0.3/-0.1/+0.1`（mAP/R1/R5/R10）；
  final相对有效B0为 `+0.9/+0.4/-0.3/-0.4`。显式relaxation在residual OFF条件下的
  matched final主效应为`+0.1 mAP`，不据此提前停止2×2；
- MR-F0 e10→e120逐参数审计通过：anchor `26/26 changed`、最大绝对差`0.00853157`、
  L2差`0.201254`，adapter `0/6 changed`且最大差/L2差均精确为零，所有张量与差值有限；
  final AMP skip累计`6`，每次均为动态scale安全回退，无NaN/Infinity/Traceback/RuntimeError/OOM；
- MR-F0 final checkpoint、runner stdout、train log SHA256依次为
  `0eb0499038af84fcbc524018cff77496321700a7e52e9bc6789ee4e65997c461`、
  `5c068ce3d76c153dce8a208b055c20fe4e58ebe2b98e1c7892b679a67d247174`、
  `319e0903d341d236b0b40f69935c41ed07539b2fe891939b51900ad58485b218`；
- GPU确认空闲、MR-P0输出目录确认不存在、exact HEAD仍为
  `ca62c475b43f17564bb09ede90de6eed53dd2d88`后，已串行启动fresh MR-P0：output=
  `log/occluded_duke/exp378_mrp0_sgd_relax_s1234`，唯一main PID=`407464`，8个子进程为
  DataLoader workers；配置确认`MODE=p0`、`ANCHOR_TRANSITION=sgd_relax`、batch `64`、
  seed `1234`、120 epochs、AMP init scale `1024`；
- MR-P0 e1 iter4出现与预检/MR-F0一致的安全AMP回退`1024→512`，随后正常训练，累计skip=`1`；
  e1 residual统计尚为零是预注册的bootstrap状态。继续监控handoff、e10/e20参数审计和matched eval。

### 2026-07-16 18:27：MR-P0 e10→e30门禁通过，转发恢复未扰动训练

- 本地`127.0.0.1:24090`转发一度消失，未触碰远端训练；经`relay4090`使用`tailscale nc`
  重新建立同一端口转发后，确认exact HEAD仍为
  `ca62c475b43f17564bb09ede90de6eed53dd2d88`，唯一main PID仍为`407464`、8个
  DataLoader workers、训练连续到e33，说明链路中断未造成训练重启或断点续跑；
- MR-P0 e10/e20/e30完整eval依次为 `36.4/46.4/61.1/67.4`、
  `47.0/58.5/71.9/77.1`、`53.1/64.9/76.9/81.8`；相对同epoch corrected hard P0依次为
  `-0.8/-0.2/-1.4/-1.5`、`+0.5/+1.4/+1.7/+0.9`、
  `+1.5/+1.6/+2.4/+2.1`（顺序为mAP/R1/R5/R10）。e20/e30早期轨迹明显为正，但仍继续到final，
  不用早期eval替代最终2×2结论；
- e10→e20逐参数审计：anchor `26/26 changed`、最大绝对差`0.00157123`、L2差
  `0.0348066`；adapter `6/6 changed`、最大绝对差`0.00121945`、L2差`0.0245855`；
  e10→e30对应为anchor `26/26`、`0.00306726`、`0.0698222`，adapter `6/6`、
  `0.00274551`、`0.0553115`，所有张量与差值有限；
- e11后pose/shape/confidence loss与teacher统计均为零，`anchor_relax_active=1`，momentum norm
  从约`0.037`衰减并稳定在约`0.025`；AMP scale已增长到`4096`，除e1安全回退外无新增skip，
  累计仍为`1`。GPU约`7.1 GiB`、利用率约`92%`，无
  NaN/Infinity/Traceback/RuntimeError/OOM，继续运行。

### 2026-07-16 18:37：MR-P0 e40/e50保持matched正向轨迹

- exact HEAD、唯一main PID `407464`与8个DataLoader workers均正常，已完成e53并运行e54；
  GPU约`7.1 GiB`、利用率约`87%`，本地转发保持在线；
- MR-P0 e40/e50完整eval分别为 `53.4/65.2/78.0/82.4`、`53.9/65.8/78.3/82.6`；
  相对同epoch corrected hard P0分别为 `+0.7/+1.3/+2.0/+1.8`、
  `+0.2/+0.7/+1.5/+1.3`（顺序为mAP/R1/R5/R10）。mAP正增益由e40的`+0.7`收窄到
  e50的`+0.2`，但四项仍全部为正，继续等待e60及后续收敛；
- e10→e50逐参数审计仍通过：anchor `26/26 changed`、最大绝对差`0.00573993`、
  L2差`0.131591`；adapter `6/6 changed`、最大绝对差`0.00527948`、L2差`0.106352`，
  所有张量与差值有限；
- e11后teacher/pose loss继续为零，anchor relaxation momentum norm约`0.025`、parameter norm
  约`24.598`；AMP scale已增长到`16384`，累计skip仍为`1`。无
  NaN/Infinity/Traceback/RuntimeError/OOM，继续运行且不作中途裁决。

### 2026-07-16 18:52：MR-P0 e60–e80 matched四项保持非负

- exact HEAD、唯一main PID `407464`与8个DataLoader workers正常，已完成e82并继续运行；
  GPU约`7.2 GiB`、利用率约`76%`，转发保持在线；
- MR-P0 e60/e70/e80完整eval依次为 `55.2/66.7/78.3/82.4`、
  `55.2/67.0/78.6/83.0`、`55.3/66.7/78.8/82.8`；相对同epoch corrected hard P0依次为
  `+0.6/+0.9/+0.2/+0.0`、`+0.2/+0.9/+0.2/+0.1`、
  `+0.4/+0.8/+1.2/+0.3`（顺序为mAP/R1/R5/R10）。e60–e80所有matched差值均非负，
  且e80 mAP仍为`+0.4`，继续等待e90/e100/final确认；
- e10→e80逐参数审计通过：anchor `26/26 changed`、最大绝对差`0.00803721`、
  L2差`0.186623`；adapter `6/6 changed`、最大绝对差`0.00752604`、L2差`0.151592`，
  所有张量与差值有限；
- e77 iter19发生一次动态AMP安全回退`131072→65536`，加上e1事件后累计skip=`2`；回退后
  loss与训练均正常。e11后teacher/pose loss仍为零，anchor relaxation momentum norm约`0.025`、
  parameter norm约`24.549`；无NaN/Infinity/Traceback/RuntimeError/OOM，继续运行。

### 2026-07-16 19:07：MR-P0 e90–e110 matched mAP持续为正

- exact HEAD、唯一main PID `407464`与8个DataLoader workers正常，已完成e111并继续最后阶段；
  GPU约`7.1 GiB`、利用率约`92%`，转发保持在线；
- MR-P0 e90/e100/e110完整eval依次为 `55.9/67.2/79.4/83.5`、
  `55.6/67.1/78.8/82.7`、`55.7/67.2/78.7/82.7`；相对同epoch corrected hard P0依次为
  `+0.2/-0.2/+0.6/+0.1`、`+0.1/+0.4/+0.6/-0.4`、
  `+0.2/+0.5/+0.4/-0.3`（顺序为mAP/R1/R5/R10）。三次后期eval的matched mAP均为正，
  R1/R5大体为正，R10在e100/e110略低；继续等待final，不用单项波动作裁决；
- e10→e110逐参数审计通过：anchor `26/26 changed`、最大绝对差`0.00864375`、
  L2差`0.201178`；adapter `6/6 changed`、最大绝对差`0.00811970`、L2差`0.163539`，
  所有张量与差值有限；
- e88 iter63发生安全AMP回退`131072→65536`，e97 iter30发生`65536→32768`，连同e1/e77
  后累计skip=`4`；当前scale已恢复到`65536`，训练与loss正常。e11后teacher/pose loss仍为零，
  anchor relaxation momentum norm约`0.025`、parameter norm约`24.536`；无
  NaN/Infinity/Traceback/RuntimeError/OOM，继续到final。

### 2026-07-16 19:22：MR-P0 final有效，hard/relax × residual OFF/ON 2×2闭合

- MR-P0已完整结束，main PID `407464`及8个workers均退出，4090显存约`2 MiB`、利用率`0%`；
  exact HEAD仍为`ca62c475b43f17564bb09ede90de6eed53dd2d88`，输出中12个
  `transformer_{10..120}.pth`齐全；final=`55.7/67.1/78.7/82.9`；
- final相对直接 matched corrected hard P0为`+0.1/+0.4/+0.3/-0.1`，相对B0为
  `+0.6/+0.4/-0.8/-0.9`（固定顺序mAP/R1/R5/R10）；e10→e120逐参数审计为anchor
  `26/26 changed`、最大绝对差`0.0086437464`、L2差`0.201177675`，adapter
  `6/6 changed`、最大绝对差`0.0081197023`、L2差`0.163539465`，所有张量与差值有限；
- 五次生产AMP安全回退依次为e1 iter4 `1024→512`、e77 iter19 `131072→65536`、
  e88 iter63 `131072→65536`、e97 iter30 `65536→32768`、e117 iter125
  `131072→65536`，fresh-process累计skip=`5`；两份日志的独立异常正则计数均为`0`；
- MR-P0 final checkpoint、runner stdout、train log SHA256依次为
  `a31a1d39e65c913ce2ad11bc2b905c2be58a0674ca9b88764383084fdf454be7`、
  `5f9569f558a6afd42aa494a07b6cff7f4bcc9a729fd86aa3ed7de1330dcbd89a`、
  `7e7360256a8d1e621bf96a0c1068fa52310f4b52b8f1bbbc658adca3f613d8af`；
- 完整同机2×2如下。hard P0−F0=`-0.3/-0.7/-0.9/-0.3`，MR-P0−MR-F0=
  `-0.3/+0.0/-0.5/-0.5`，说明当前geometry residual在两种transition下均无益；
  MR-F0−hard F0=`+0.1/-0.3/-0.1/+0.1`、MR-P0−hard P0=
  `+0.1/+0.4/+0.3/-0.1`，说明显式relaxation只有约`+0.1 mAP`的matched小主效应；

| anchor transition | residual OFF | residual ON | ON−OFF |
|---|---:|---:|---:|
| hard | F0 `55.9/67.4/79.3/83.3` | P0 `55.6/66.7/78.4/83.0` | `-0.3/-0.7/-0.9/-0.3` |
| explicit relaxation | MR-F0 `56.0/67.1/79.2/83.4` | MR-P0 `55.7/67.1/78.7/82.9` | `-0.3/+0.0/-0.5/-0.5` |

- 当前证据支持保留的是residual-OFF的内生姿态场配置：hard F0−B0=
  `+0.8/+0.7/-0.2/-0.5`，MR-F0−B0=`+0.9/+0.4/-0.3/-0.4`。它尚不能单独把收益
  归因于“正确关节语义”，因为仍混合了bootstrap课程、内部anchor、Gaussian renderer与PSG；
  需要同机R0/D0和joint/wrong-pose语义审计继续拆分；
- 下一执行顺序登记为：先做final checkpoint的外部pose `correct/shuffle/None` exact parity、
  joint permutation、错误姿态/常量场与stage关闭等低成本语义审计；随后在同一4090串行运行
  fresh D0（持续teacher、residual OFF）→J0（持续teacher、residual ON）→R0（外部target
  ViTPose PSG）。D0/J0用于闭合原预注册pose-supervision × residual 2×2，R0提供外部姿态实用
  上限；最小Gate B保留RG0、N0/置换bootstrap与最佳residual-OFF checkpoint语义审计。
  只有J0−D0重新显示residual正贡献时才恢复U0/C0；不得据此停止TAPF。

### 2026-07-16 19:40：同机fresh D0通过门禁并串行启动

- 4090 exact HEAD=`ca62c475b43f17564bb09ede90de6eed53dd2d88`，tracked工作树clean；
  D0与hard F0 config逐行diff只有注释、`POSE_TAPF_MODE: f0→d0`和独立OUTPUT_DIR；batch=`64`、
  seed=`1234`、120 epochs及其余配方一致；
- PyTorch `1.13.1+cu117` / CUDA `11.7`原生D0 e11 batch64 CUDA preflight通过：
  `TAPF_CUDA_PREFLIGHT_PASS`、10-step runtime parity、真实overflow `128→64`、AMP tracker
  `0/1/0 total=1`；pose objective只更新anchor，adapter gradient/delta均为零，ReID不更新
  anchor，符合持续pose监督+residual OFF定义；
- 确认GPU空闲、D0输出目录不存在后启动fresh同机D0。当前唯一main PID=`442976`，8个子进程
  均为DataLoader workers，GPU约`7.3 GiB`、利用率约`81%`；已完成e1并进入e2，e1 pose loss
  约`2.92→2.91`、anchor confidence约`0.54`，residual统计为零；e1 iter4出现一次预期的安全
  AMP回退`1024→512`，累计skip=`1`；无NaN/Inf/Traceback/RuntimeError/OOM；
- 启动壳层最初把自己的PID写到根目录`/wrapper.pid`，未影响训练；发现后未重启进程，已删除
  错位文件，并把实际Python main PID `442976`写回正确OUTPUT的`wrapper.pid`。后续以该PID监控；
  D0完整结束前禁止启动J0/R0或任何其他4090训练。

### 2026-07-16 19:47：同机D0 e10 handoff与持续监督正常

- exact HEAD仍为`ca62c475b43f17564bb09ede90de6eed53dd2d88`；唯一main PID=`442976`、
  8个DataLoader workers正常，已完成e13并运行e14；GPU训练期约`7.3 GiB`，无重复训练；
- D0 e10完整eval=`36.1/45.0/60.3/66.7`，相对同epoch corrected hard F0为
  `-1.7/-2.6/-2.6/-2.7`（mAP/R1/R5/R10）。这是bootstrap/handoff端点且`<e60`，只记录
  早期随机轨迹，不作性能负裁决；
- e10 `student_fraction=1.0`，e11后D0继续读取teacher并保持pose loss约`0.94→0.86`，
  anchor confidence约`0.77→0.81`、sigma约`0.148→0.140`；geometry shift/log-scale与adapter
  更新保持为零，符合D0定义；
- AMP除e1 iter4的安全`1024→512`外无新增skip，当前scale=`1024`、累计skip=`1`；两份日志
  异常正则计数均为`0`。本地转发失活时已按预案仅经relay4090/tailscale nc恢复，远端main与
  epoch连续，未重启训练。

### 2026-07-16 19:52：同机D0 e20持续监督与参数归属门禁通过

- D0 e20完整eval=`45.3/55.6/69.8/75.3`，相对同epoch corrected hard F0为
  `-1.0/-1.7/-2.3/-1.5`（mAP/R1/R5/R10）。当前仍`<e60`，只记录持续监督臂的早期负向
  轨迹，不作性能裁决；
- e10→e20逐参数审计：anchor `26/26 changed`、最大绝对差`0.06298754`、L2差
  `1.37894551`；geometry adapter `0/6 changed`、最大差与L2差均精确为零，所有差值有限。
  这符合D0后期只由pose objective持续更新anchor、residual OFF的预注册定义；
- e20前pose loss约`0.75`、anchor confidence约`0.83`、sigma约`0.131`，teacher继续读取，
  shift/log-scale均为零；AMP累计skip仍为`1`且日志无异常。继续完整运行，不提前启动J0/R0。

### 2026-07-16 20:07：同机D0 e30/e40回到matched混合轻正轨迹

- exact HEAD与main PID `442976`不变，唯一main及8个DataLoader workers正常；已完成e48并
  运行e49，GPU约`7.4 GiB`、利用率约`87%`；转发重建未触碰远端训练，epoch连续；
- D0 e30/e40完整eval分别为`52.2/64.0/76.3/80.5`、`53.2/63.9/77.6/81.9`；相对
  同epoch corrected hard F0分别为`+0.0/+0.5/+0.6/+0.0`、
  `+0.2/-0.3/+1.0/+0.3`（mAP/R1/R5/R10）。e10/e20的早期负差已收敛为混合轻正，
  但仍`<e60`，不作性能裁决；
- 持续pose loss约降至`0.50`，anchor confidence约`0.836`、sigma约`0.107`，teacher保持
  有限读取；geometry shift/log-scale持续为零。AMP scale已正常增长至`16384`，除e1外无新增
  skip、累计仍`1`；两份日志异常计数均为`0`。继续到e50/e60与final。

### 2026-07-16 20:22：同机D0 e50–e70保持matched混合轨迹

- exact HEAD与main PID `442976`不变，唯一main及8个workers正常；已完成e74，GPU约
  `7.4 GiB`、利用率约`85%`，训练连续；
- D0 e50/e60/e70完整eval依次为`53.4/64.5/78.0/81.9`、
  `55.6/67.1/79.3/82.7`、`55.4/66.2/79.5/83.5`；相对同epoch corrected hard F0依次为
  `-0.7/-1.0/-0.4/-1.1`、`+0.4/-0.2/+0.4/-0.2`、
  `-0.2/-1.4/-0.1/-0.5`（mAP/R1/R5/R10）。e60 mAP一度为`+0.4`，e70回到`-0.2`，
  尚无稳定单向主效应；不以单点停止，继续到final；
- pose loss约`0.48`、anchor confidence约`0.835`、sigma约`0.105`，teacher保持有限读取，
  geometry shift/log-scale仍为零；e67 iter71出现一次动态AMP安全回退`65536→32768`，连同e1
  后累计skip=`2`，回退后loss/训练正常；日志异常计数仍为`0`。

### 2026-07-16 20:37：同机D0 e90/e100 matched四项转为一致正向

- exact HEAD与main PID `442976`不变，唯一main及8个workers正常；e100已完整eval并运行e101+，
  GPU约`7.4 GiB`、利用率约`88%`；
- D0 e80/e90/e100完整eval依次为`55.5/66.1/79.0/82.7`、
  `56.3/67.6/79.8/83.5`、`56.1/67.4/80.0/83.3`；相对同epoch corrected hard F0依次为
  `+0.0/-0.3/-0.2/-0.3`、`+0.5/+0.4/+0.3/+0.2`、
  `+0.4/+0.3/+0.8/+0.1`（mAP/R1/R5/R10）。e90/e100连续两次四项全正，提示持续
  pose supervision可能优于hard freeze，但仍需e110/final复核，不能用后期单点定案；
- pose loss约`0.46`、anchor confidence约`0.836`、sigma约`0.103`，teacher持续有限读取，
  geometry shift/log-scale保持零；e76/e86/e95分别发生三次动态AMP安全回退，加e1/e67后累计
  skip=`5`，每次回退后训练连续且loss有限；两份日志异常计数仍为`0`。

### 2026-07-16 20:54：同机D0 final有效，持续姿态监督带来matched小正收益

- D0已完整结束，main PID `442976`及8个DataLoader workers均退出，4090显存约`2 MiB`、
  利用率`0%`；exact HEAD仍为`ca62c475b43f17564bb09ede90de6eed53dd2d88`，tracked工作树
  无改动，输出中12个`transformer_{10..120}.pth`齐全；e110/final依次为
  `56.1/67.5/79.7/83.3`、`56.2/67.6/79.8/83.4`；
- e110相对同epoch corrected hard F0为`+0.2/+0.0/+0.3/-0.1`；final相对直接matched hard F0为
  `+0.3/+0.2/+0.5/+0.1`，相对B0为`+1.1/+0.9/+0.3/-0.4`（固定顺序
  mAP/R1/R5/R10）。相对3090 D0 final为`+0.5/+0.0/+0.8/+0.2`，但该项只作跨机趋势；
  单seed证据支持持续姿态监督对hard freeze有小幅正贡献，尚不能称稳定或显著；
- e10→e120逐参数审计通过：anchor `26/26 changed`、最大绝对差`0.219773680`、L2差
  `3.938361915`；geometry adapter `0/6 changed`且最大差/L2差均为零，所有张量与差值有限，
  符合D0持续pose objective更新anchor、residual OFF的定义；final训练统计中pose loss=`0.467`、
  anchor confidence=`0.836`、sigma=`0.104`，shift/log-scale仍为零；
- 七次生产AMP安全回退依次为e1 iter4 `1024→512`、e67 iter71、e76 iter184、e86 iter83、
  e95 iter207、e105 iter90、e115 iter207的`65536→32768`，fresh-process累计skip=`7`；
  两份日志的NaN/Inf/Traceback/RuntimeError/OOM严格正则计数均为`0`；
- final checkpoint、runner stdout、train log SHA256依次为
  `ba03262a40ddea6b346c3f2587225ece67a6cf7a52368786cf14a227787e195c`、
  `8b98cf858dd061b0c92d2d0952c8c83f87596e2064a8dcac870988b325ca881b`、
  `c179a13aef61f1e757012a32f1250ca316e98c4a84d07c229bb144ebb694a34a`；GPU空闲且J0输出
  不存在，下一步只做J0 e11 batch64 CUDA preflight与fresh启动，禁止并行R0。

### 2026-07-16 21:00：fresh同机J0通过门禁并串行启动

- exact HEAD仍为`ca62c475b43f17564bb09ede90de6eed53dd2d88`且tracked工作树无改动；J0与D0
  config逐行diff只有注释、`POSE_TAPF_MODE: d0→j0`和独立OUTPUT_DIR，batch=`64`、
  seed=`1234`、120 epochs及其余配方一致；启动前J0输出不存在、GPU无compute进程；
- PyTorch `1.13.1+cu117` / CUDA `11.7`原生J0 e11 batch64 CUDA preflight通过：
  `TAPF_CUDA_PREFLIGHT_PASS`、10-step runtime parity、真实overflow `128→64`、AMP tracker
  `0/1/0 total=1`；pose objective只更新anchor（grad约`8.22e-1`），不流入adapter/backbone；
  ReID objective不更新anchor，但同时到达adapter output/upstream（约`1.10e-3`/`2.59e-10`）
  与backbone。单步anchor/adapter delta约`1.55e-3`/`3.75e-6`，符合持续pose监督+
  ReID geometry adaptation的J0定义；
- 在D0进程完整退出且GPU空闲后启动fresh J0，OUTPUT为
  `log/occluded_duke/exp378_j0_joint_control_s1234`；当前唯一main PID=`478136`、8个子进程
  均为DataLoader workers，GPU约`7.3 GiB`、利用率约`92%`。已进入e1，teacher/bootstrap正常，
  shift/log-scale按预期仍为零；e1 iter4发生一次安全AMP回退`1024→512`、累计skip=`1`；
  runner日志无NaN/Inf/Traceback/RuntimeError/OOM。J0完整结束前禁止启动R0或其他4090训练。

### 2026-07-16 21:07：同机J0 e10 handoff正常，e11后双目标持续

- exact HEAD与main PID `478136`不变，唯一main及8个DataLoader workers正常；已完成e13并
  运行e14，GPU约`7.5 GiB`、利用率约`84%`，无重复4090训练；
- J0 e10完整eval=`36.5/46.5/61.9/68.4`，相对同epoch fresh D0为
  `+0.4/+1.5/+1.6/+1.7`（mAP/R1/R5/R10）。这是bootstrap/handoff端点且`<e60`，
  geometry residual尚未形成可解释幅度，只记录随机轨迹，不作性能裁决；
- e11后teacher继续读取且pose loss约`0.86`、anchor confidence约`0.81`、sigma约`0.14`，
  符合J0持续pose supervision；geometry shift/log-scale当前日志三位小数仍显示`0.000`，
  需等待e20 checkpoint逐参数确认adapter `6/6`有限更新，不能据日志舍入值误判未学动；
- AMP除e1 iter4的安全`1024→512`外无新增skip，当前scale=`1024`、累计skip=`1`；两份日志
  的NaN/Inf/Traceback/RuntimeError/OOM严格正则计数均为`0`，继续完整运行。

### 2026-07-16 21:23：同机J0 e20–e40 residual确实更新，matched早中期为正

- exact HEAD与main PID `478136`不变，唯一main及8个workers正常；已完成e41并运行e42，
  GPU约`7.5 GiB`、利用率约`88%`，训练连续；
- J0 e20/e30/e40完整eval依次为`46.1/57.2/70.4/75.1`、
  `52.7/63.8/77.0/81.8`、`53.8/65.1/78.5/82.8`；相对同epoch fresh D0依次为
  `+0.8/+1.6/+0.6/-0.2`、`+0.5/-0.2/+0.7/+1.3`、
  `+0.6/+1.2/+0.9/+0.9`（mAP/R1/R5/R10）。mAP连续三次为正，e40四项全正，
  但仍属早中期单seed轨迹，继续到e60/final后再判断`J0−D0`；
- e10→e20逐参数审计通过：anchor `26/26 changed`（max`0.06563297`、L2`1.392474`），
  adapter `6/6 changed`（max`0.00121945`、L2`0.0245774`）；e10→e40进一步达到anchor
  `26/26`（max`0.16818881`、L2`2.984773`）和adapter `6/6`（max`0.00409448`、
  L2`0.0824892`），所有张量与差值有限。由此确认geometry residual确实在学习，日志三位小数
  的shift/log-scale `0.000`只是舍入，不能解释为adapter未更新；
- e40 pose loss约`0.53`、anchor confidence约`0.84`、sigma约`0.11`，teacher持续有限读取；
  AMP仍只有e1一次安全回退、累计skip=`1`，当前scale=`8192`；两份日志异常正则计数均为`0`。

### 2026-07-16 21:37：同机J0 e50/e60回到matched混合轨迹

- exact HEAD、唯一main PID `478136`及8个DataLoader workers正常，已完成e66并运行e67；
  GPU约`7.5 GiB`、利用率约`79%`，无并行4090训练；
- J0 e50/e60完整eval分别为`52.3/64.0/76.7/81.3`、`55.8/67.6/78.9/83.1`；
  相对同epoch fresh D0分别为`-1.1/-0.5/-1.3/-0.6`、
  `+0.2/+0.5/-0.4/+0.4`（mAP/R1/R5/R10）。e50全面偏低、e60 mAP/R1/R10转正但
  R5仍低，说明e20–e40的正向轨迹尚未稳定；继续到final，不提前恢复U0/C0；
- e10→e60逐参数审计通过：anchor `26/26 changed`（max`0.20134598`、L2`3.548772`），
  adapter `6/6 changed`（max`0.00623930`、L2`0.125682`），所有张量与差值有限；持续
  teacher读取、pose loss约`0.48`、anchor confidence约`0.84`、sigma约`0.105`，J0双目标正常；
- e67 iter71发生第二次动态AMP安全回退`65536→32768`，加上e1后累计skip=`2`；回退后loss
  与训练正常，两份日志异常正则计数仍为`0`。

### 2026-07-16 21:53：同机J0 e80/e90 matched四项恢复小正

- exact HEAD、唯一main PID `478136`及8个workers正常，已完成e93并运行e94；GPU约
  `7.5 GiB`、利用率约`88%`，无并行训练；
- J0 e70/e80/e90完整eval依次为`55.4/67.0/78.7/83.1`、
  `55.8/66.7/79.2/83.3`、`56.4/68.1/79.9/83.8`；相对同epoch fresh D0依次为
  `+0.0/+0.8/-0.8/-0.4`、`+0.3/+0.6/+0.2/+0.6`、
  `+0.1/+0.5/+0.1/+0.3`（mAP/R1/R5/R10）。e70仍混合，e80/e90四项均小幅为正；
  这是当前residual在持续pose supervision下可能有正贡献的后期信号，但幅度小且单seed，
  必须等待e100/e110/final确认后才决定是否恢复U0/C0；
- e10→e90逐参数审计通过：anchor `26/26 changed`（max`0.21840245`、L2`3.881616`），
  adapter `6/6 changed`（max`0.00785422`、L2`0.158213`），所有张量与差值有限；teacher持续
  读取，pose loss约`0.47`、anchor confidence约`0.84`、sigma约`0.104`；
- e76 iter184发生`65536→32768`、e79 iter209发生`32768→16384`安全AMP回退，连同e1/e67
  后fresh-process累计skip=`4`；当前scale=`32768`，回退后训练/loss正常，两份日志异常正则
  计数均为`0`。

### 2026-07-16 22:09：同机J0 final有效，residual对mAP无可分辨正贡献

- J0已完整结束，main PID `478136`及8个workers均退出，4090显存约`2 MiB`、利用率`0%`；
  exact HEAD仍为`ca62c475b43f17564bb09ede90de6eed53dd2d88`且tracked工作树无改动，12个
  `transformer_{10..120}.pth`齐全；e100/e110/final依次为`56.1/67.7/79.3/83.8`、
  `56.2/68.0/79.5/83.8`、`56.2/67.9/79.5/83.9`；
- 相对同epoch fresh D0，e100/e110/final依次为`+0.0/+0.3/-0.7/+0.5`、
  `+0.1/+0.5/-0.2/+0.5`、`+0.0/+0.3/-0.3/+0.5`（mAP/R1/R5/R10）。final相对
  hard P0为`+0.6/+1.2/+1.1/+0.9`，相对B0为`+1.1/+1.2/+0.0/+0.1`；后两项混合了
  持续pose supervision与residual，不能替代直接`J0−D0`归因；
- `J0−D0` final的mAP为`+0.0`，R1/R10小正而R5为负，当前单seed证据不支持geometry
  residual带来可分辨mAP正贡献，因此不恢复U0/C0；这只停止当前residual小变体，不停止TAPF，
  后续继续R0、RG0与pose语义审计；
- e10→e120逐参数审计通过：anchor `26/26 changed`（max`0.22069210`、L2`3.932768`），
  adapter `6/6 changed`（max`0.00811923`、L2`0.163524`），所有张量与差值有限；final pose
  loss=`0.466`、anchor confidence=`0.836`、sigma=`0.104`，teacher持续有限读取；
- 七次AMP安全回退为e1/e67/e76/e79/e98/e108/e118，fresh-process累计skip=`7`，final
  scale=`32768`；两份日志NaN/Inf/Traceback/RuntimeError/OOM严格正则计数均为`0`；final
  checkpoint、runner stdout、train log SHA256依次为
  `ee42da549c0abc397b9180875b7dd546ad46c74f05bacd8289072c119474b94e`、
  `00012988f5e8cc8d9fd869661c3202e7e41c7bba79ab5d77a624e33f293c5ba7`、
  `aabced0fcbbad87c3d20eab9665ea12bb9089f30549cc149c8f92ab4d6cef248`。

### 2026-07-16 22:13：fresh同机R0通过外部姿态门禁并串行启动

- exact HEAD仍为`ca62c475b43f17564bb09ede90de6eed53dd2d88`且tracked工作树无改动；R0与B0
  config逐行diff只有注释、`POSE_PSG_STAGES: []→[-1]`和独立OUTPUT_DIR，batch=`64`、
  seed=`1234`、120 epochs、target-person pose loader及其余配方一致；R0定义为固定外部
  ViTPose target heatmap经Stage-3 PSG，`POSE_TAPF=False`；
- 由于TAPF专用preflight明确拒绝`POSE_TAPF=False`，改用同等生产组件执行R0独立batch64
  CUDA forward/backward门禁并通过：`R0_CUDA_PREFLIGHT_PASS`；第一批实际送入`s3_b0` PSG的
  raw heatmap与`person-0 × person_mask`逐位相等，范围`[-0.008530,1.019228]`且有限；
  loss=`17.75963020`，PSG末层grad=`2.54465`、单步delta=`0.00254777`，AMP scale
  `1024→1024`，证明external target pose路径与PSG优化真实连通；
- 确认J0进程完整退出、GPU空闲且R0输出不存在后，启动fresh R0，OUTPUT为
  `log/occluded_duke/exp378_r0_external_teacher_s1234`；当前唯一main PID=`512553`、8个子进程
  均为DataLoader workers，GPU约`7.1 GiB`、利用率约`86%`，已进入e1；runner日志无
  NaN/Inf/Traceback/RuntimeError/OOM。R0完整结束前禁止启动RG0或其他4090训练。

### 2026-07-16 22:23：同机R0 e10/e20相对B0早期明显为正

- exact HEAD与main PID `512553`不变，唯一main及8个DataLoader workers正常；已完成e20并
  进入后续阶段，GPU训练期约`7.1 GiB`，无并行4090训练；
- R0 e10/e20完整eval分别为`37.7/47.6/62.2/68.6`、
  `46.7/56.5/71.0/76.8`；相对同epoch B0分别为`+0.8/+0.6/+0.4/+0.3`、
  `+4.4/+2.8/+5.3/+6.0`（mAP/R1/R5/R10）。e20差值很大，但当前仍`<e60`且两臂
  早期优化轨迹可不同，只记录external pose PSG的收敛速度，不作final性能结论；
- e10→e20逐参数审计通过：Stage-3 PSG `8/8 changed`、最大绝对差`0.15209773`、L2差
  `0.37431206`，所有参数与差值有限，确认固定external target heatmap持续驱动PSG更新；
- R0使用标准GradScaler路径，不输出TAPF专用scale/skip tracker；当前无AMP/nonfinite警告，
  两份日志NaN/Inf/Traceback/RuntimeError/OOM严格正则计数均为`0`，继续到e60/final。

### 2026-07-16 22:38：同机R0 e30/e40相对B0继续为正

- exact HEAD、唯一main PID `512553`及8个DataLoader workers正常，已完成e47并运行e48；
  GPU约`7.1 GiB`、利用率约`88%`，训练连续且无并行arm；
- R0 e30/e40完整eval分别为`52.4/63.8/75.9/81.0`、
  `54.1/65.3/78.6/82.4`；相对同epoch B0分别为`+1.8/+1.9/+0.7/+0.6`、
  `+1.1/+0.2/+1.0/+0.5`（mAP/R1/R5/R10）。所有差值仍为正，但当前`<e60`，
  继续到后期/final确认，不把收敛速度直接解释为最终优势；
- e10→e40 Stage-3 PSG `8/8 changed`、最大绝对差`0.23435822`、L2差`0.55313484`，
  所有参数与差值有限；无AMP/nonfinite warning，两份日志异常严格正则计数仍为`0`。

### 2026-07-16 22:53：同机R0 e50–e70 matched mAP保持约+1.1

- exact HEAD、唯一main PID `512553`及8个workers正常，已完成e74并运行e75；GPU约
  `7.1 GiB`、利用率约`81%`，训练连续；
- R0 e50/e60/e70完整eval依次为`53.3/64.8/77.5/82.7`、
  `54.9/66.2/78.2/82.5`、`55.5/66.8/79.0/83.4`；相对同epoch B0依次为
  `+1.2/+1.3/+0.6/+1.7`、`+1.1/+1.0/+0.5/+0.5`、
  `+1.1/+0.4/+0.1/+0.3`（mAP/R1/R5/R10）。e50–e70 matched mAP稳定在约`+1.1`，
  但仍需final后与F0/MR-F0/D0/J0作系统级比较；R0依赖外部pose推理，不能与RGB-only TAPF
  直接合并为同一实用成本；
- e10→e70 Stage-3 PSG `8/8 changed`、最大绝对差`0.25942323`、L2差`0.60992942`，
  所有参数与差值有限；无AMP/nonfinite warning，两份日志异常严格正则计数均为`0`。

### 2026-07-16 23:08：同机R0 e80–e100 matched mAP保持+0.9至+1.2

- exact HEAD、唯一main PID `512553`及8个workers正常，已完成e101并运行e102；GPU约
  `7.1 GiB`、利用率约`79%`，训练连续；
- R0 e80/e90/e100完整eval依次为`55.5/66.6/79.0/83.3`、
  `56.0/67.4/79.6/84.1`、`56.0/66.7/79.3/83.7`；相对同epoch B0依次为
  `+0.9/+0.0/+0.1/+0.2`、`+1.1/+1.0/+0.2/+0.9`、
  `+1.2/+0.1/+0.0/+0.4`（mAP/R1/R5/R10）。后期matched mAP继续保持`+0.9∼+1.2`，
  说明external target-pose PSG优势不是只存在于早期收敛；仍等待e110/final确认；
- e10→e100 Stage-3 PSG `8/8 changed`、最大绝对差`0.26431412`、L2差`0.62164573`，
  所有参数与差值有限；无AMP/nonfinite warning，两份日志异常严格正则计数均为`0`。

### 2026-07-16 23:24：同机R0 final有效，RGB-only D0匹配external pose上界

- 本地`24090`转发一度失活，仅经relay4090/tailscale nc恢复监控链路；远端R0早已完整结束，
  未重启或修改训练。main PID `512553`及8个workers均退出，4090显存约`2 MiB`、利用率`0%`；
  exact HEAD仍为`ca62c475b43f17564bb09ede90de6eed53dd2d88`且tracked工作树无改动，12个
  `transformer_{10..120}.pth`齐全；e110/final分别为`56.1/67.4/79.4/83.8`、
  `56.1/67.4/79.5/83.7`；
- R0 final相对B0为`+1.0/+0.7/+0.0/-0.1`，相对hard F0为
  `+0.2/+0.0/+0.2/+0.4`，相对MR-F0为`+0.1/+0.3/+0.3/+0.3`；相对D0为
  `-0.1/-0.2/-0.3/+0.3`，相对J0为`-0.1/-0.5/+0.0/-0.2`（固定顺序
  mAP/R1/R5/R10）；
- R0证明固定external target ViTPose PSG能带来约`+1.0 mAP`，但RGB-only推理的D0在mAP上
  反而高`0.1`且前三项均不低。这是单seed系统级证据：支持“内部姿态场可以匹配外部姿态依赖”
  的候选叙事，不足以声称显著优于R0；raw teacher与Gaussian renderer仍由RG0拆分；
- e10→e120 Stage-3 PSG `8/8 changed`、最大绝对差`0.26467100`、L2差`0.62249891`，
  所有参数与差值有限；全程无AMP/nonfinite warning，两份日志NaN/Inf/Traceback/
  RuntimeError/OOM严格正则计数均为`0`；final checkpoint、runner stdout、train log SHA256
  依次为`586f05f45f2612998b172412764e73a3a048839b62d4d14ee4f654c260dd8834`、
  `73246f9189e81ce34fe1eca469aa65236fffa1349c7d6a5e07a8ec57998e6ebe`、
  `7298b1bcf64ac50de640c429ad0599b238e53c7f2bdc512536af81d980267d1c`；
- 当前exact commit没有RG0 config，GPU保持空闲；先完成“raw ViTPose→同一17关节Gaussian renderer”
  的最小单变量设计/代码/config与CUDA exact-parity门禁，生成新exact commit/bundle后才可fresh启动，
  禁止用R0输出续训或现场改运行资产。

### 2026-07-16 23:58：RG0 最小实现与本地门禁通过，远端生产门禁前不启动

- RG0已按预注册单变量实现：R0继续使用external target-person raw heatmap；RG0对同一person-0
  heatmap/score/mask执行positive clamp、mass normalization、diagonal moments、`[0.025,0.25]`
  sigma clamp及confidence×peak-normalized Gaussian，再进入同一Stage-3 PSG；不实例化anchor、
  geometry adapter、handoff或pose loss；
- 默认`POSE_EXTERNAL_FIELD_RENDERER=raw`不实例化module、不增加parameter/persistent buffer且不消耗
  RNG；full-model CPU门禁证明R0/RG0初始state、构造RNG与optimizer groups逐位一致，person-0 field
  exact，PSG在resize后只执行一次sigmoid：`RG0_MODEL_INVARIANTS_PASS`；
- 原TAPF 12项unit加RG0 5项unit共`17/17 PASS`；另以当前父commit中的旧TAPF源码作冻结oracle，
  对F0/D0/P0/J0、train/eval、e1/e6/e10/e11比较state、构造RNG、field、pose loss与全部stats，
  得到`TAPF_PRE_REFACTOR_EXACT_PARITY_PASS`；
- 两名独立Codex子代理完成静态实现与证据链审查，均未发现代码层阻塞bug；按审查补充了full-data
  audit、生产batch64 CUDA/AMP snapshot、sigma饱和/zero-mass/score越界等统计、LOG_PERIOD采样及
  RG0 fail-closed组合门禁。审查记录见`codex_rg0_review.md`，全程未使用Claude；
- 当前4090仍为空闲（约`2 MiB/0%`），旧exact HEAD仍为
  `ca62c475b43f17564bb09ede90de6eed53dd2d88`。正式启动仍阻塞于四项远端证据：全量真实数据、
  PyTorch1.13 batch64 AMP、旧R0跨commit exact snapshot、旧TAPF checkpoint/CUDA parity；任何一项
  未通过都不得启动RG0。

### 2026-07-17 00:39：RG0 四类4090生产门禁全部通过，进入最终execution固化

- RG0 execution candidate以旧生产exact
  `ca62c475b43f17564bb09ede90de6eed53dd2d88`为直接父提交，保留其PyTorch1.13与TAPF AMP
  审计代码；当前门禁HEAD=`3f789f06c494b179f597ac96485284c48d31d11a`，独立repo为
  `/home/afr/SOLIDER-REID-exp378-rg0-b126ef2`，未从本地分叉文档branch覆盖生产代码；
- full-data门禁`RG0_FULL_DATA_AUDIT_PASS`：cache train/query/gallery分别
  `15618/2210/17661`样本，cache与实际dataset输出的nonfinite、
  positive-confidence-zero-mass均为`0`；train全部`441`个pad/crop位置共
  `117088146`个joint状态、其中`115694118`个active状态全部安全，最小active positive
  mass=`0.12665433`。解析式与真实bilinear求和的最大float32误差为`2.2888e-5`；首轮
  `1e-5`容差只因此误报，随后仅把诊断容差改为`1e-4`并完整重跑，renderer/config未变；
- 数据域不是空对照：actual raw negative fraction约`0.2507–0.3326`，score越界比例约
  `0.234%–0.461%`；RG0确实执行positive clamp、confidence clamp与Gaussianization。
  active sigma上界命中率最高约`2.17%`、下界命中率最高约`3.53%`，不存在全体sigma饱和；
- PyTorch`1.13.1+cu117`原生门禁通过：17项unit全部PASS，CUDA full-model得到
  `TAPF_MODEL_INVARIANTS_PASS / RG0_MODEL_INVARIANTS_PASS`；MR-F0/MR-P0 e11 batch64均通过
  10-step legacy parity、真实AMP overflow`128→64`、无anchor objective gradient及matched
  delta语义，分别为anchor/adapter
  `9.9204e-4/0`与`8.9284e-4/3.7462e-6`；
- 共享renderer回归通过：父提交旧源码与candidate在F0/P0/D0/J0、hard/relax、
  train/eval、e1/e6/e10/e11共48个CUDA/autocast情形逐位一致；
  MR-F0/MR-P0真实final checkpoint各`259`个tensor strict-load成功，checkpoint SHA256分别为
  `0eb0499038af84fcbc524018cff77496321700a7e52e9bc6789ee4e65997c461`与
  `a31a1d39e65c913ce2ad11bc2b905c2be58a0674ca9b88764383084fdf454be7`；
- 旧R0与candidate R0跨commit硬比较为`reference_exact_match=true`：生产batch、初始state、
  optimizer、descriptor、featmaps、loss、181个gradient tensor、target heatmap/score、两处PSG
  field/resize/sigmoid及一步update全部SHA逐位相等；共享batch SHA=`c8eea2a045f3…25a58`，
  初始state=`dd1cfd5082ee…ac01`，after-step state=`ecbe3de08554…6f2e`；
- RG0同一batch64 AMP preflight PASS：field为float32 `64×17×96×32`，单次PSG sigmoid exact，
  AMP scale `1024→1024`且无skip，PSG final grad norm=`3.57427`、一步delta=`0.00358151`，
  peak memory=`5825.73 MiB`；batch内active mass min=`12.9609`、sigma-x/y范围分别
  `[0.06428,0.25]/[0.03181,0.17806]`、rendered peak-confidence误差为`0`；
- 原始证据位于`remote_artifacts/exp378_rg0_preflight_b126ef2/4090/`。关键SHA256：
  full-data log=`24c8908269c97127607363ec432083034a45c1676578ee7722ef0ba1d283d2fe`，
  TAPF regression log=`ab3c615d62bc2821a9699f944694886d28083d9b10be0e73e5f294a0e361eecd`，
  R0 exact JSON=`017a4ffe550ce5dfa61a7c84154ba3e7ac771382e7be518835a8d6bfa1398889`，
  RG0 JSON=`99029660cb29c9be17977d933ef381b576c4c9cffe080e5ea8f7c976cae92c27`；
- 当前GPU=`2 MiB/0%`、无训练主进程，RG0 output不存在。下一步只提交本段monitor证据并生成
  最终exact execution commit/full bundle；再次核对HEAD、bundle、config、空output和GPU后才fresh
  启动RG0，不恢复或重复任何已完成arm。

### 2026-07-17 01:00：RG0 最终execution复验通过并fresh串行启动

- 最终exact execution commit为
  `1fe1e32734f6dafdf1c6b12b24950e585e918868`，相对门禁HEAD只增加上述`monitor.md`证据；
  RG0 config SHA256仍为
  `d8d19aedb4529113946916d5606b40f672293b253f5b1fdb0c288098be2d09cf`。完整history bundle
  `remote_artifacts/exp378_rg0_full_1fe1e32.bundle`已由`git bundle verify`确认无prerequisite，
  SHA256=`44f82f9af4db04ab01e40d24abc92ea6dee374d50d310ca6343cf8c26b48c57c`；经relay上传
  在`6.8 MiB`处停滞后已明确改名为`.partial`，没有冒充有效bundle。远端实际部署使用只含最后
  doc commit的增量bundle，SHA256=`8170d2769b40c3461f303e12cdba33b8d094388e7b8a9c364428965587d6c359`，
  从`3f789f0`严格fast-forward到最终HEAD；
- 最终HEAD重新执行R0跨commit snapshot，`reference_exact_match=true`：相对旧生产
  `ca62c475b43f17564bb09ede90de6eed53dd2d88`的batch、initial/after-step state、optimizer、
  descriptor/featmaps/loss、181个gradient tensor、target heatmap/score及两处PSG输入仍逐位相等；
  随后的RG0 batch64 CUDA/AMP preflight再次PASS，field为float32`64×17×96×32`，AMP
  `1024→1024`且无skip，PSG grad norm=`3.57427`、一步delta=`0.00358151`、peak memory
  `5825.73 MiB`，zero-mass与rendered peak-confidence error均为`0`；
- 在最终HEAD、tracked clean、GPU=`2 MiB/0%`、无compute/train进程且OUTPUT不存在的原子门禁下，
  fresh启动`log/occluded_duke/exp378_rg0_external_gaussian_s1234`；唯一main PID=`556835`，
  `wrapper.pid`一致，8个子进程均为DataLoader workers。启动后GPU约`7126 MiB`，训练已健康进入
  e4；e1采样中raw negative fraction约`0.313–0.319`、confidence约`0.818–0.834`、
  sigma-x/y约`[0.063,0.25]/[0.030,0.216]`，active positive mass始终正，near-zero mass及
  rendered peak-confidence error均为`0`；
- 两次启动前壳层曾因`pgrep`匹配到同一条尚未执行的远端命令字符串而fail-closed，均发生在创建
  OUTPUT之前；现场复核无进程、无GPU占用且目录不存在，因此不是训练启动/失败/重跑。改用进程
  `comm`与参数联合检查后只执行了一次真实fresh启动。当前runner/train日志的NaN、独立词边界
  Inf、Traceback、RuntimeError及OOM计数均为`0`；继续串行监控，`<e60`不作性能负裁决，
  e10→e20必须验证Stage-3 PSG参数有限变化。

### 2026-07-17 01:03：RG0 e8运行健康，修正automation中的旧R0状态

- exact HEAD仍为`1fe1e32734f6dafdf1c6b12b24950e585e918868`且tracked clean；唯一main
  PID=`556835`与8个DataLoader workers正常，GPU约`7126 MiB/90%`，已完整结束e7并运行e8；
  尚未产生首次e10 eval，因此本次不填性能差值，也不作性能判断；
- e7–e8采样中raw negative fraction约`0.301–0.329`、confidence约`0.817–0.861`，
  active positive mass最小值保持正；sigma-x/y仍在预注册`[0.025,0.25]`边界内，边界命中比例低，
  near-zero mass与rendered peak-confidence error持续为`0`。严格词边界NaN/Inf及Traceback、
  RuntimeError、OOM计数均为`0`；
- RG0的`POSE_TAPF=False`，当前processor只对`POSE_TAPF=True`臂在训练日志中输出实时
  `tapf_amp_scale/skip/total`，因此不能从RG0运行日志声称实时scale或累计skip；现有可报告证据仅为
  最终batch64 preflight的`1024→1024`无skip，以及截至e8没有nonfinite/overflow warning。
  为保持R0/RG0代码路径matched且禁止修改运行中代码，本臂不现场补日志；把这一可观测性边界保留
  在最终解释中；
- heartbeat automation此前仍写着已完成R0的旧HEAD/PID/OUTPUT，现已更新为RG0 exact commit、
  bundle/config SHA、PID、matched R0差值口径及RG0结束后的N0/语义审计顺序，防止后续误重启R0。

### 2026-07-17 01:18：RG0完成e10/e20/e30 matched评估并通过PSG轨迹审计

- exact HEAD与config SHA仍分别为`1fe1e32734f6dafdf1c6b12b24950e585e918868`和
  `d8d19aedb4529113946916d5606b40f672293b253f5b1fdb0c288098be2d09cf`，tracked clean；
  唯一main PID=`556835`及8个DataLoader workers正常，GPU约`7148 MiB/88%`，已完成e33并
  运行e34；严格NaN/Inf、Traceback、RuntimeError、OOM、AMP skip/overflow warning均为`0`；
- RG0 e10/e20/e30完整评估依次为`38.6/49.3/64.6/71.0`、
  `47.1/57.9/71.9/77.6`、`52.2/63.9/76.6/81.5`。由本次日志数字现场减去同epoch R0后，
  四项差值依次为`+0.9/+1.7/+2.4/+2.4`、`+0.4/+1.4/+0.9/+0.8`、
  `-0.2/+0.1/+0.7/+0.5`（固定顺序mAP/R1/R5/R10）；早期全正信号在e30收敛为混合，
  `<e60`不作性能负裁决，继续到final；
- e10→e20 checkpoint逐参数审计通过：Stage-3 PSG canonical参数`8/8 changed`，最大绝对差
  `0.163722336`、聚合L2差`0.359994698`，两端参数与全部差值有限；state_dict中的8个
  `psg_modules.*`别名与`psg_modules_dict.s3_*`逐位一致。符合RG0只训练同一Stage-3 PSG的定义；
- e33–e34的external field仍有真实数值域变换：raw negative fraction约`0.316–0.325`，
  confidence约`0.836–0.843`，active positive mass保持正，sigma边界命中比例低，near-zero mass与
  rendered peak-confidence error持续为`0`。没有修改运行中代码/config或重启训练。

### 2026-07-17 01:33：RG0 e60四项matched正收益，继续等待后期与final

- exact HEAD/config SHA与main PID仍为`1fe1e32734f6dafdf1c6b12b24950e585e918868`、
  `d8d19aedb4529113946916d5606b40f672293b253f5b1fdb0c288098be2d09cf`、`556835`；
  tracked clean，唯一main与8个DataLoader workers健康，GPU约`7164 MiB/88%`，已完成e61并
  运行e62；严格NaN/Inf、Traceback、RuntimeError、OOM、AMP skip/overflow warning均为`0`；
- RG0 e40/e50/e60完整评估为`54.1/65.7/77.9/82.8`、`54.4/66.0/78.7/83.1`、
  `55.4/66.9/79.1/83.4`；由日志现场减去同epoch R0后分别为
  `+0.0/+0.4/-0.7/+0.4`、`+1.1/+1.2/+1.2/+0.4`、`+0.5/+0.7/+0.9/+0.9`
  （mAP/R1/R5/R10）。e50/e60连续四项全正且e60 mAP为`+0.5`，给Gaussianization机制提供
  正向燃料；仍是单seed中期轨迹，继续到e70–final，不把e60单点升级为最终结论；
- e10→e60 Stage-3 PSG canonical参数`8/8 changed`，最大绝对差`0.273737252`、聚合L2差
  `0.588762251`，两端参数和差值全部有限，8个state_dict别名逐位一致；更新轨迹符合RG0定义；
- e61–e62 external field的raw negative fraction约`0.311–0.320`，confidence约`0.828–0.843`，
  active positive mass始终为正，sigma保持有界；near-zero mass与rendered peak-confidence error
  仍为`0`。未修改运行中代码/config或重启训练。

### 2026-07-17 01:49：RG0 e70/e80持续正向，e90轨迹混合但mAP仍正

- exact HEAD/config SHA、main PID与tracked状态均未改变；唯一main及8个DataLoader workers健康，
  e90完整评估已落盘，GPU约`7118 MiB`。截至e90严格NaN/Inf、Traceback、RuntimeError、OOM、
  AMP skip/overflow warning均为`0`；
- RG0 e70/e80/e90为`55.9/67.0/80.2/84.0`、`56.0/67.4/79.8/84.0`、
  `56.3/67.3/80.1/83.9`；现场减去同epoch R0后依次为
  `+0.4/+0.2/+1.2/+0.6`、`+0.5/+0.8/+0.8/+0.7`、`+0.3/-0.1/+0.5/-0.2`
  （mAP/R1/R5/R10）。e70/e80连续四项全正，e90 mAP仍为`+0.3`但R1/R10轻微转负；证据支持
  Gaussianization在中后期不弱于raw field的候选判断，仍需e100/e110/final闭合；
- e10→e90 Stage-3 PSG canonical参数`8/8 changed`，最大绝对差`0.282272041`、聚合L2差
  `0.607080350`，两端参数/差值有限且8个state_dict别名逐位一致；
- e89–e90 raw negative fraction约`0.312–0.318`、confidence约`0.828–0.843`，active positive mass
  始终为正，sigma有界，near-zero mass与rendered peak-confidence error仍为`0`。继续到final，
  不因e90单项小负提前裁决。

### 2026-07-17 02:08：RG0完整结束并通过final/全轨迹审计

- exact execution commit=`1fe1e32734f6dafdf1c6b12b24950e585e918868`、full history bundle
  SHA256=`44f82f9af4db04ab01e40d24abc92ea6dee374d50d310ca6343cf8c26b48c57c`、config
  SHA256=`d8d19aedb4529113946916d5606b40f672293b253f5b1fdb0c288098be2d09cf`保持不变；原main
  PID=`556835`及workers均已退出，GPU=`2 MiB/0%`，output内`transformer_10.pth`至
  `transformer_120.pth`共12个checkpoint齐全；未续训、重启或手工跳epoch；
- RG0 e100/e110/e120完整评估分别为`56.2/67.1/79.8/84.0`、
  `56.2/67.0/80.0/83.8`、`56.2/66.9/79.8/83.9`；由日志现场减去同epoch R0后依次为
  `+0.2/+0.4/+0.5/+0.3`、`+0.1/-0.4/+0.6/+0.0`、`+0.1/-0.5/+0.3/+0.2`
  （mAP/R1/R5/R10）；final相对B0、hard F0、MR-F0、D0、J0依次为
  `+1.1/+0.2/+0.3/+0.1`、`+0.3/-0.5/+0.5/+0.6`、`+0.2/-0.2/+0.6/+0.5`、
  `+0.0/-0.7/+0.0/+0.5`、`+0.0/-1.0/+0.3/+0.0`；
- 12个checkpoint的Stage-3 PSG全轨迹审计通过：canonical参数始终为8个，所有张量有限，
  11组相邻checkpoint均为`8/8 changed`，e10→e120为`8/8 changed`，最大绝对差
  `0.283552587`、聚合L2差`0.609875201`，所有`psg_modules.*`别名与canonical参数逐位一致：
  `RG0_PSG_FULL_TRAJECTORY_PASS`；
- final checkpoint、runner stdout、train log的SHA256依次为
  `3553b5053b21c29f454b20b95623ba308b3c144008a00a35aa536472ba70f081`、
  `f67fd50f3b314fd3bde1ac62f5c3b176910ad8214cdadcfb5413ea91e8875158`、
  `85eb4a5b7fcbbdced1a62675a6a5065e20f21bc7a30760926c46bdba5e47e42a`；全程严格
  NaN/Inf、Traceback、RuntimeError、OOM、nonfinite/overflow warning均为`0`。因
  `POSE_TAPF=False`，生产processor没有实时scaler/累计skip字段；这里只保留最终preflight
  `1024→1024`无skip及生产日志无warning的可观测证据，不虚构累计AMP skip；
- **当前判断**：RG0相对raw R0仅`+0.1 mAP`，Gaussianization没有损伤PSG收益，但也没有可分辨
  的独立mAP增益；D0、RG0同为`56.2 mAP`，R0为`56.1 mAP`，说明内部姿态场、外部Gaussian
  与外部raw pose在本单seed中大体同档。该结果不终止TAPF，也不恢复已无mAP贡献的U0/C0。
  4090现已空闲；下一步先把N0固定为严格matched的置换bootstrap单变量对照，完成独立
  design/config/output/preflight门禁后才允许fresh启动，当前尚未启动N0。

### 2026-07-17 02:31：N0固定置换定义与本地实现门禁通过，4090生产门禁前不启动

- N0已明确以corrected residual-OFF hard F0为唯一直接对照，不再使用会混入初始化/训练难度的
  随机controller：bootstrap期间以固定destination→source 17-cycle
  `[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,0]`同步重排target-person teacher heatmap与
  joint confidence；保留空间支持/置信度多重集、anchor/renderer/PSG、hard transition、batch64、
  seed1234与120epoch配方，geometry residual保持OFF。定义commit=`d3a9010`；
- candidate implementation commit=`6535981`：新增默认空列表config开关、N0独立config、TAPF内部
  exactly-once置换、active/fixed-point日志及fail-closed preflight；未新增参数或persistent state，
  N0 config相对F0除注释外只多置换列表与独立output；
- 本地uv CPU单元`15/15 PASS`，覆盖非法列表拒绝、heatmap/score同步置换、内部置换与显式外部
  relabeling在e1/e10逐位等价、e1 field exact、pose梯度隔离、F0/N0 state/参数/optimizer一致；
  full-model CPU得到`TAPF_MODEL_INVARIANTS_PASS / RG0_MODEL_INVARIANTS_PASS /
  N0_MODEL_INVARIANTS_PASS`，F0/N0构造RNG、完整state、production optimizer groups exact，N0
  Stage-3 PSG输入为一次置换teacher；
- 审查见`codex_n0_review.md`：本地实现PASS，但不得冒充4090 PyTorch1.13.1证据。当前4090仍无
  训练，N0未启动；下一步在独立新repo以真实依赖重跑full-model invariants、15项unit、e1/e11
  batch64 CUDA/AMP、真实overflow与teacher-off门禁，全部通过并固化exact commit/bundle/config
  SHA、确认output不存在后才允许fresh串行启动。

### 2026-07-17 02:45：N0 4090 candidate门禁PASS，等待final exact execution复验

- 首个candidate单元运行在PyTorch1.13 restricted loader读取测试自产生的optimizer-state内存包时
  fail-fast；当时未进入N0 CUDA forward、未启动训练。兼容修复commit=`8aa4738`只对该可信测试包
  增加显式fallback，不改变生产checkpoint加载或N0实现；随后从新bundle、新repo重新开始全部门禁；
- 4090 candidate exact=`8aa473898921da100608dee501f6dad489fc59b5`，full-history bundle
  SHA256=`3e91e7269c649555b574500ad47bbca29bef6a7b23ad1c2bb38bdd515a9f476b`，config
  SHA256=`50b516f78458bc08c8d0d0192d561934facbeb938eddcd30d03f93b41b090814`；独立repo=
  `/home/afr/SOLIDER-REID-exp378-n0-preflight-8aa4738`，tracked clean、N0 output不存在；
- 原生`torch=1.13.1+cu117`下`15/15 unit PASS`，CUDA full-model得到
  `TAPF_MODEL_INVARIANTS_PASS / RG0_MODEL_INVARIANTS_PASS / N0_MODEL_INVARIANTS_PASS`；F0/N0
  完整state、构造RNG、production optimizer groups exact，e1 PSG raw field为exact once-permuted
  target-person teacher；
- e1/e11 batch64 CUDA/AMP均`TAPF_CUDA_PREFLIGHT_PASS`，每次10-step legacy parity与真实overflow
  `128→64`通过。e1 pose loss有限、anchor delta=`9.72232689e-4`、adapter delta=`0`，pose只更新
  anchor；e11 pose=`None`，anchor/adapter gradient与delta均为`0`，hard-freeze且teacher完全不读；
  两次均通过eval correct/None/不可索引external pose exact parity；
- 四份原始日志已回传`remote_artifacts/exp378_n0_preflight_8aa4738/4090/`，unit/full-model/e1/e11
  SHA256依次为`9918315890e27fba94ebf54eed055792e2a592eba4b790936c6b84df2c2de53d`、
  `859e1cd9871b45d21f6b9274770b6643302858e1a76b8a1b5fb1a2e1c962e62f`、
  `c249b2601afd9b576e03925bdb57507dea487283fb6acaed53bc77059b8a40ca`、
  `685a4bfe4a4015ddabc952b1793eb6c82baa32bb3aa162d360727ccb64632190`；门禁后无训练主进程、
  GPU=`2 MiB/0%`；
- 当前仍不启动N0。下一步只提交本证据，生成新的exact execution commit/full bundle与独立final
  repo，复验HEAD/config、15项unit、full-model和e1/e11 CUDA关键门禁；final repo全部通过且output
  仍不存在后才允许fresh启动。

### 2026-07-17 02:55：N0 final生产门禁全PASS并fresh串行启动

- final exact execution commit=`1bae8243daec40a9892cd237862737b6ba328afd`，full-history bundle
  SHA256=`7208ce641a04238f877ded6e9b59e40464d6017d39451b212b406de1a02def00`，config
  `configs/occluded_duke/exp378_n0_permuted_bootstrap.yml` SHA256=
  `50b516f78458bc08c8d0d0192d561934facbeb938eddcd30d03f93b41b090814`；独立生产repo=
  `/home/afr/SOLIDER-REID-exp378-n0-1bae824`，output=
  `log/occluded_duke/exp378_n0_permuted_bootstrap_s1234`；
- final repo在原生`torch=1.13.1+cu117`再次通过`15/15 unit`、
  `TAPF_MODEL_INVARIANTS_PASS / RG0_MODEL_INVARIANTS_PASS / N0_MODEL_INVARIANTS_PASS`、e1/e11
  batch64 CUDA/AMP、两次10-step legacy parity及两次真实overflow `128→64`；e1 anchor delta=
  `9.7223e-4`且adapter delta=`0`，e11 teacher完全不读、pose=`None`、anchor/adapter gradient与
  delta均为`0`，correct/None/不可索引external pose descriptor parity通过；
- final四份门禁日志已回传`remote_artifacts/exp378_n0_final_1bae824/4090/`，unit/full-model/e1/e11
  SHA256依次为`9918315890e27fba94ebf54eed055792e2a592eba4b790936c6b84df2c2de53d`、
  `24fd6bb84cf4faac2407e3754b093a7cf8095eed90647f81c82906bb5b2cb606`、
  `f7359e181b6fa0d7bf08aaf6b06f5691669bba9fe3ce23a05bdc466d73927b37`、
  `304a6ff9059d00eb972ea81b19d0bdfc2be1d3bd67ca22fc197cfcfedd0beb4b`；
- output不存在且GPU空闲的原子门禁通过后只fresh启动一次，main PID=`601979`。首个启动命令曾被
  `pgrep -f`匹配命令自身而fail-closed，未创建output且无训练进程；改用按executable name筛选后
  才执行唯一一次真实启动，不属于重复训练；
- N0直接对照固定为corrected residual-OFF hard F0：`MODE=f0`、hard transition、geometry
  residual OFF，仅把bootstrap target-person teacher heatmap与joint confidence同步执行固定
  destination→source 17-cycle `[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,0]`；N0≈F0只允许解释
  为正确解剖通道名称不必要，不能解释为空间pose-like support无效。

### 2026-07-17 02:56：N0 e1–e7运行健康，等待e10完整eval

- 远端HEAD仍精确为`1bae8243daec40a9892cd237862737b6ba328afd`，tracked diff为空（仅预期的
  untracked `data/`与`remote_artifacts/`）；唯一main PID=`601979`及8个DataLoader workers，GPU=
  `7422 MiB / 81% / 48°C`；
- e1–e7均完整结束，单epoch约`29.8–30.7s`，最新ETA约`56min`；bootstrap pose loss有限并由
  e1末约`2.94`下降到e7末约`1.14`，student fraction按日程由`0.0`升至`0.4`；
- 所有采样点`teacher_permutation_active=1`、`teacher_permutation_fixed_points=0`，teacher heatmap与
  confidence保持同步固定derangement；`shift_rms/log_scale_rms=0/0`，geometry adapter保持关闭；
- runner/train日志中严格`NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow`异常为0。该生产
  processor不输出实时AMP scale/skip，因此只保留preflight真实overflow证据，不虚构生产累计skip；
- 当前没有完整eval，继续到e10并现场计算相对matched corrected hard F0 e10
  `37.8/47.6/62.9/69.4`的mAP/R1/R5/R10四项显式差值；`<e60`不作性能负裁决。

### 2026-07-17 02:58：N0 e10完整eval与e11 hard-freeze切换均健康

- e10日志现场读取mAP/R1/R5/R10=`37.3/46.7/62.6/68.5`；相对同epoch corrected hard F0
  `37.8/47.6/62.9/69.4`的差值为`-0.5/-0.9/-0.3/-0.9`。这是bootstrap端点且`<e60`，只记录
  trajectory，不作性能负裁决；
- e10全程`teacher_permutation_active=1`、fixed points=`0`、student fraction=`1.0`、geometry
  shift/log-scale=`0/0`；pose loss有限，e10末约`0.96`；
- e11已健康越过handoff：`tapf_pose/shape_loss/confidence_loss=0/0/0`，teacher raw/sigmoid/
  confidence/valid-fraction日志字段均为`0`，anchor field保持有限，geometry shift/log-scale仍为`0/0`。
  final生产门禁已证明e11 teacher完全不读且anchor/adapter硬冻结；生产日志当前与该预期一致；
- 继续等待e20 checkpoint后做e10→e20逐参数审计；预期anchor `0/26`、adapter `0/6` changed，且
  所有张量有限。当前不重启、不改运行中代码/config。

### 2026-07-17 03:16：N0 e20/e30/e40与hard-freeze逐参数审计均PASS

- e20=`46.6/57.5/71.5/77.0`，相对同epoch corrected hard F0
  `46.3/57.3/72.1/76.8`为`+0.3/+0.2/-0.6/+0.2`；
- e30=`52.9/64.8/77.0/81.4`，相对同epoch corrected hard F0
  `52.2/63.5/75.7/80.5`为`+0.7/+1.3/+1.3/+0.9`；
- e40=`54.0/66.1/78.4/82.0`，相对同epoch corrected hard F0
  `53.0/64.2/76.6/81.6`为`+1.0/+1.9/+1.8/+0.4`；轨迹从e20混合转为e30/e40四项全正，
  但三点均`<e60`，继续到final，不提前作性能裁决；
- checkpoint逐参数审计：e10→e20、e10→e30、e10→e40均为anchor `0/26` changed、adapter
  `0/6` changed，max delta=`0`、L2=`0`且全部张量有限；hard-freeze与residual OFF严格成立；
- 最新检查运行至e47：远端exact HEAD不变，tracked diff为空（仅预期untracked数据目录），唯一
  main PID=`601979`及8 workers，GPU约`7114 MiB/87%/53°C`；e11后pose/teacher统计持续为`0`、
  shift/log-scale持续为`0/0`，runner/train严格异常计数为`0`。

### 2026-07-17 03:30：N0 e50/e60/e70轨迹反转，机制审计持续PASS

- e50=`53.9/66.0/78.6/82.9`，相对同epoch corrected hard F0
  `54.1/65.5/78.4/83.0`为`-0.2/+0.5/+0.2/-0.1`；
- e60=`55.4/67.8/79.1/83.2`，相对同epoch corrected hard F0
  `55.2/67.3/78.9/82.9`为`+0.2/+0.5/+0.2/+0.3`；
- e70=`55.5/66.7/78.6/83.1`，相对同epoch corrected hard F0
  `55.6/67.6/79.6/84.0`为`-0.1/-0.9/-1.0/-0.9`；e60四项小正随后e70四项转负，明确不能
  依据e60单点定案，继续到e80/e90/final；
- e10→e50、e10→e60、e10→e70逐参数均为anchor `0/26` changed、adapter `0/6` changed，
  max delta=`0`、L2=`0`且全部有限；hard-freeze/residual OFF持续严格成立；
- 最新运行至e73+，exact HEAD、唯一main+8 workers与GPU均健康；e11后pose/teacher统计持续为`0`、
  shift/log-scale持续为`0/0`，生产processor无实时AMP字段，日志严格异常计数仍为`0`。

### 2026-07-17 03:45：N0 e80/e90/e100后期三点均非负，继续等待final

- e80=`55.6/66.9/79.2/83.3`，相对同epoch corrected hard F0
  `55.5/66.4/79.2/83.0`为`+0.1/+0.5/+0.0/+0.3`；
- e90=`56.2/67.9/79.9/83.3`，相对同epoch corrected hard F0
  `55.8/67.2/79.5/83.3`为`+0.4/+0.7/+0.4/+0.0`；
- e100=`56.0/67.5/79.6/83.3`，相对同epoch corrected hard F0
  `55.7/67.1/79.2/83.2`为`+0.3/+0.4/+0.4/+0.1`；e80–e100连续三点四项均非负，但仍等待
  e110/final后再定性，不用局部轨迹代替final；
- e10→e80、e10→e90、e10→e100逐参数均为anchor `0/26`、adapter `0/6` changed，max
  delta=`0`、L2=`0`且全部有限；hard-freeze/residual OFF继续严格成立；
- 最新运行至e102+，exact HEAD、唯一main+8 workers、GPU与日志均健康；e11后pose/teacher统计和
  geometry residual持续为零，严格异常计数仍为`0`。

### 2026-07-17 04:00：N0完整结束并通过final/全轨迹审计

- e110=`56.1/67.6/80.0/83.5`，相对同epoch corrected hard F0
  `55.9/67.5/79.4/83.4`为`+0.2/+0.1/+0.6/+0.1`；final e120=
  `56.1/67.6/80.0/83.4`，相对matched hard F0=`+0.2/+0.2/+0.7/+0.1`，相对B0=
  `+1.0/+0.9/+0.5/-0.4`；额外相对D0=`-0.1/+0.0/+0.2/+0.0`、R0=
  `+0.0/+0.2/+0.5/-0.3`、RG0=`-0.1/+0.7/+0.2/-0.5`；
- N0单seed与hard F0没有可分辨负差，支持“当前机制不依赖正确的17关节通道名称”这一有限判断；
  但N0保留同一人的pose-like空间支持与confidence多重集，不能解释为空间姿态支持无效，后续语义审计
  仍是必要归因；
- 原main PID=`601979`及8 workers均已自然退出，GPU=`2 MiB/0%/33°C`；远端exact HEAD仍为
  `1bae8243daec40a9892cd237862737b6ba328afd`，tracked diff为空（仅预期untracked数据目录）；
- 12个checkpoint齐全。以e10为基准审计全部e10–e120 checkpoint：anchor全程`0/26` changed、
  adapter全程`0/6` changed，max delta=`0`，全部张量有限；hard-freeze/residual OFF全轨迹PASS；
- final checkpoint SHA256=`f9ed5a65285e38f0f723a2afa34ce641c27c6d931ef7b6cd6d6b56c7b086b582`，
  runner SHA256=`00af13a01b00a8738c17886cf5d99cba045472d969b06e5598c4b2cdb8d443f0`，train log
  SHA256=`864d5f9fe4d8c5733be1d00b10c32bec4e9c73706d4c68eb168a5749771f39fd`；严格
  `NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow`计数为`0`。生产processor无实时AMP
  scale/skip字段，因此只保留preflight真实overflow证据，不虚构生产累计skip；
- N0禁止重启/续训。下一步只推进最佳residual-OFF checkpoint的只读语义审计：correct/shuffle/None
  external pose exact parity、joint permutation、错误姿态/常量场退化、Stage-3 PSG关闭、teacher
  agreement、flip equivariance与通道占用；不启动新训练、不触发H0。

### 2026-07-17 04:32：D0 e90冻结语义审计完整结束，当前单锚点PSG不支持姿态因果叙事

- 审计对象严格固定为事后选择的D0 e90 checkpoint，原始指标=`56.2984/67.6471/79.8190/83.5294`；
  它只用于同checkpoint配对干预，不替代D0 final=`56.2/67.6/79.8/83.4`。独立远端repo exact
  commit=`7955c8155f9db1cb337a2d907384172da144bd64`，完整bundle SHA256=
  `9e19327ee30cb487da5247c6a129d6c06bf561b9761918cb01493ee6270486d0`；D0 e90 checkpoint
  SHA256=`c5407d30d145b92c1995b137ea917187bfb5c1e7c04cd662a44362ae68b4c253`；
- 4090原生PyTorch 1.13.1的12项CPU utility测试与batch64单batch全12臂门禁均PASS。正式审计覆盖
  `19,871`张query+gallery图，correct-start/end四项指标和descriptor逐位相同；external
  correct/shuffle/None/unindexable四臂也与correct逐位相同，严格证明TAPF部署态完全不读取外部
  `pose_dict`；
- 冻结反事实相对correct的mAP/R1/R5/R10显式百分点差值为：matched-wrong-field=
  `-0.0155/+0.0000/+0.0000/+0.0000`，joint-permutation=
  `+0.0024/+0.0000/+0.0000/-0.0453`，confidence-permutation=
  `+0.0002/+0.0000/+0.0000/+0.0000`，spatial-constant=
  `-0.0238/-0.0905/+0.0000/-0.0453`，zero-field=
  `-0.0536/-0.0452/+0.0000/+0.0452`。所有mAP绝对差均`<0.1`，按预注册区间均为无可分辨贡献；
- 真正绕过全部Stage-3 PSG后，指标降为`53.6154/63.9367/76.2896/80.4525`，相对correct=
  `-2.6829/-3.7104/-3.5294/-3.0769`。因此检索明确依赖训练后的PSG模块，却几乎不依赖其输入场的
  图像对应、17通道名称、confidence或空间结构。`zero_field`仍会在PSG内部经过`sigmoid(0)=0.5`
  与已学习encoder，不能等价为模块关闭；当前最一致解释是静态/容量性重标定，而非姿态场因果贡献；
- 内生anchor本身仍呈pose-like统计：teacher-confidence `>=0.3`的normalized coordinate error=
  `0.07647`、pseudo-PCK@0.05=`0.55392`、posterior cosine=`0.82764`；flip对齐后的posterior
  cosine=`0.94674`、coordinate error=`0.03043`；17通道winner全部占用，归一化occupancy entropy=
  `0.96320`。这说明“中间场像姿态”与“检索真正使用姿态语义”是两件事，前者不能替代后者证据；
- 所有field干预和PSG bypass均实际改变descriptor；所有短生命周期hook均恢复；模型state SHA在
  全12臂前后逐位相同，严格异常计数为零。原formal PID=`641396`及wrapper已退出，GPU回到
  `2 MiB/0%`，远端tracked clean；formal JSON/log SHA256分别为
  `0e0282b72ae0d15133b1426e13917d9ed9b9b693d17d981dca748f07839db04b` /
  `2e1ad96f10e87d6b9f103f1d3c3db29cff6b41f6eb338b798e84d5e5478797ae`，原始证据已回传
  `remote_artifacts/exp378_semantic_audit_7955c81/4090/`；
- 当前单锚点TAPF/PSG的强姿态因果叙事据此不成立，不补该实现的multi-seed、ResNet或Video迁移，
  也不触发H0训练。TAPF问题对象不因一个机制失败而停止，但Hierarchical版本必须先独立重设计
  **null-separable field consumer**（null field严格恒等）与parameter-matched RGB/static controls，
  并把逐层field干预敏感性作为训练后的硬门禁，不能把多个现有PSG堆叠包装成逐层方法。

## Eval 记录

每次完整 eval 均在此追加 mAP/R1/R5/R10，不只记录最好结果。

> exp378归因臂的备注必须同时写matched直接对照差值：MR-F0比较 corrected hard F0，MR-P0比较
> corrected hard P0，顺序固定为mAP/R1/R5/R10；final另写相对B0。跨epoch或跨机差值不得代替
> matched同机同epoch对照。

> 参数审计后的有效性覆盖：旧 P0 的 e20–e120与旧 F0 的 e20–e40均标记为
> `INVALID_AS_HARD_FREEZE / VALID_RELAXATION_PILOT`。表中数值不得进入论文正式结果表或
> hard-freeze门禁，但保留为显式 relaxation 配对实验的机制先导证据。

| 机器 | arm | epoch | mAP | R1 | R5 | R10 | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| 4090 | P0 | 10 | 36.5 | 45.0 | 61.9 | 68.2 | predicted-only；handoff 已完成 |
| 3090 | D0 | 10 | 37.9 | 47.4 | 63.7 | 69.1 | predicted-only；跨机只作趋势 |
| 4090 | P0 | 20 | 46.2 | 56.8 | 71.2 | 76.2 | teacher/pose loss 已关闭 |
| 4090 | P0 | 30 | 51.8 | 63.3 | 75.3 | 80.5 | teacher/pose loss 已关闭 |
| 4090 | P0 | 40 | 53.4 | 64.8 | 76.9 | 81.4 | teacher/pose loss 已关闭 |
| 4090 | P0 | 50 | 54.0 | 65.4 | 78.4 | 82.9 | teacher/pose loss 已关闭 |
| 4090 | P0 | 60 | 55.7 | 67.3 | 79.3 | 83.5 | `CONTINUE_WITH_FUEL`；等待 exact B0 |
| 3090 | D0 | 20 | 46.1 | 56.9 | 71.1 | 76.7 | 持续 pose supervision；跨机只作趋势 |
| 4090 | P0 | 70 | 55.3 | 66.6 | 79.4 | 83.4 | teacher/pose loss 已关闭 |
| 4090 | P0 | 80 | 55.7 | 67.1 | 79.0 | 83.4 | teacher/pose loss 已关闭 |
| 4090 | P0 | 90 | 56.0 | 67.3 | 79.5 | 83.4 | teacher/pose loss 已关闭 |
| 3090 | D0 | 30 | 52.5 | 64.3 | 76.7 | 81.4 | 持续 pose supervision；跨机只作趋势 |
| 4090 | P0 | 100 | 56.0 | 67.6 | 79.4 | 83.3 | teacher/pose loss 已关闭 |
| 4090 | P0 | 110 | 56.1 | 67.7 | 79.5 | 83.7 | teacher/pose loss 已关闭 |
| 4090 | P0 | 120 | 56.2 | 67.8 | 79.6 | 83.7 | P0完整结束；等待 exact controls |
| 3090 | D0 | 40 | 54.0 | 65.2 | 77.4 | 82.4 | 持续 pose supervision；跨机只作趋势 |
| 3090 | D0 | 50 | 54.4 | 66.1 | 78.4 | 82.5 | 持续 pose supervision；跨机只作趋势 |
| 4090 | B0 | 10 | 36.9 | 47.0 | 61.8 | 68.3 | exact clean B0；同配方对照 |
| 4090 | B0 | 20 | 42.3 | 53.7 | 65.7 | 70.8 | exact clean B0；同配方对照 |
| 4090 | B0 | 30 | 50.6 | 61.9 | 75.2 | 80.4 | exact clean B0；P0−B0 mAP `+1.2` |
| 4090 | B0 | 40 | 53.0 | 65.1 | 77.6 | 81.9 | exact clean B0；P0−B0 mAP `+0.4` |
| 4090 | B0 | 50 | 52.1 | 63.5 | 76.9 | 81.0 | exact clean B0；P0−B0 mAP `+1.9` |
| 4090 | B0 | 60 | 53.8 | 65.2 | 77.7 | 82.0 | exact clean B0；P0−B0 mAP `+1.9` |
| 3090 | D0 | 60 | 55.1 | 66.5 | 78.1 | 82.5 | 持续 pose supervision；跨机只作趋势 |
| 4090 | B0 | 70 | 54.4 | 66.4 | 78.9 | 83.1 | exact clean B0；P0−B0 mAP `+0.9` |
| 3090 | D0 | 70 | 55.2 | 66.7 | 78.8 | 82.7 | 持续 pose supervision；跨机只作趋势 |
| 4090 | B0 | 80 | 54.6 | 66.6 | 78.9 | 83.1 | exact clean B0；P0−B0 mAP `+1.1` |
| 3090 | D0 | 80 | 55.4 | 66.6 | 78.7 | 83.2 | 持续 pose supervision；跨机只作趋势 |
| 4090 | B0 | 90 | 54.9 | 66.4 | 79.4 | 83.2 | exact clean B0；P0−B0 mAP `+1.1` |
| 3090 | D0 | 90 | 55.9 | 67.4 | 79.5 | 83.3 | 持续 pose supervision；跨机只作趋势 |
| 4090 | B0 | 100 | 54.8 | 66.6 | 79.3 | 83.3 | exact clean B0；P0−B0 mAP `+1.2` |
| 3090 | D0 | 100 | 55.6 | 66.9 | 78.7 | 83.1 | 持续 pose supervision；跨机只作趋势 |
| 4090 | B0 | 110 | 55.1 | 66.8 | 79.5 | 83.6 | exact clean B0；P0−B0 mAP `+1.0` |
| 3090 | D0 | 110 | 55.7 | 67.6 | 79.0 | 83.3 | 持续 pose supervision；跨机只作趋势 |
| 4090 | B0 | 120 | 55.1 | 66.7 | 79.5 | 83.8 | exact final；P0−B0 mAP/R1 `+1.1/+1.1` |
| 3090 | D0 | 120 | 55.7 | 67.6 | 79.0 | 83.2 | 跨机 final；正式归因待4090同机 D0 |
| 4090 | F0 | 10 | 37.6 | 46.7 | 63.0 | 69.5 | bootstrap端点；后续 arm 因 anchor漂移无效 |
| 4090 | F0 | 20 | 47.6 | 58.1 | 71.9 | 77.2 | `INVALID_MOMENTUM_DRIFT`；仅诊断 |
| 4090 | F0 | 30 | 51.9 | 63.3 | 75.5 | 80.5 | `INVALID_MOMENTUM_DRIFT`；仅诊断 |
| 4090 | F0 | 40 | 54.3 | 65.7 | 79.2 | 82.9 | `INVALID_MOMENTUM_DRIFT`；仅诊断；e45终止 |
| 4090 | P0-fix | 10 | 37.2 | 46.6 | 62.5 | 68.9 | hard-freeze execution；bootstrap端点 |
| 4090 | P0-fix | 20 | 46.5 | 57.1 | 70.2 | 76.2 | anchor e10→e20逐位不变；adapter `6/6`更新 |
| 4090 | P0-fix | 30 | 51.6 | 63.3 | 74.5 | 79.7 | anchor e10→e30逐位不变；`<e60`不裁决 |
| 4090 | P0-fix | 40 | 52.7 | 63.9 | 76.0 | 80.6 | hard-freeze execution；继续 |
| 4090 | P0-fix | 50 | 53.7 | 65.1 | 76.8 | 81.3 | hard-freeze execution；继续 |
| 4090 | P0-fix | 60 | 54.6 | 65.8 | 78.1 | 82.4 | 对有效 B0 e60为 `+0.8/+0.6/+0.4/+0.4` |
| 4090 | P0-fix | 70 | 55.0 | 66.1 | 78.4 | 82.9 | 对有效 B0 e70 mAP/R1为 `+0.6/-0.3` |
| 4090 | P0-fix | 80 | 54.9 | 65.9 | 77.6 | 82.5 | 对有效 B0 e80 mAP/R1为 `+0.3/-0.7` |
| 4090 | P0-fix | 90 | 55.7 | 67.4 | 78.8 | 83.4 | 对有效 B0 e90 mAP/R1为 `+0.8/+1.0` |
| 4090 | P0-fix | 100 | 55.5 | 66.7 | 78.2 | 83.1 | 对有效 B0 e100 mAP/R1为 `+0.7/+0.1` |
| 4090 | P0-fix | 110 | 55.5 | 66.7 | 78.3 | 83.0 | 对有效 B0 e110 mAP/R1为 `+0.4/-0.1` |
| 4090 | P0-fix | 120 | 55.6 | 66.7 | 78.4 | 83.0 | final；P0−B0=`+0.5/+0.0/-1.1/-0.8`；继续归因 |
| 4090 | F0-fix | 10 | 37.8 | 47.6 | 62.9 | 69.4 | corrected hard-freeze；handoff端点 |
| 4090 | F0-fix | 20 | 46.3 | 57.3 | 72.1 | 76.8 | anchor `0/26`、adapter `0/6` changed |
| 4090 | F0-fix | 30 | 52.2 | 63.5 | 75.7 | 80.5 | corrected hard-freeze；继续 |
| 4090 | F0-fix | 40 | 53.0 | 64.2 | 76.6 | 81.6 | corrected hard-freeze；继续 |
| 4090 | F0-fix | 50 | 54.1 | 65.5 | 78.4 | 83.0 | hard P0−F0 mAP/R1=`-0.4/-0.4` |
| 4090 | F0-fix | 60 | 55.2 | 67.3 | 78.9 | 82.9 | hard P0−F0 mAP/R1=`-0.6/-1.5`；不提前停止 |
| 4090 | F0-fix | 70 | 55.6 | 67.6 | 79.6 | 84.0 | hard P0−F0 mAP/R1=`-0.6/-1.5`；继续 |
| 4090 | F0-fix | 80 | 55.5 | 66.4 | 79.2 | 83.0 | hard P0−F0 mAP/R1=`-0.6/-0.5`；继续 |
| 4090 | F0-fix | 90 | 55.8 | 67.2 | 79.5 | 83.3 | hard P0−F0 mAP/R1=`-0.1/+0.2`；继续 |
| 4090 | F0-fix | 100 | 55.7 | 67.1 | 79.2 | 83.2 | hard P0−F0 mAP/R1=`-0.2/-0.4`；等待 final |
| 4090 | F0-fix | 110 | 55.9 | 67.5 | 79.4 | 83.4 | corrected hard-freeze；等待 final |
| 4090 | F0-fix | 120 | 55.9 | 67.4 | 79.3 | 83.3 | final；F0−B0=`+0.8/+0.7/-0.2/-0.5`，P0−F0=`-0.3/-0.7/-0.9/-0.3` |
| 4090 | MR-F0 | 10 | 37.9 | 47.1 | 63.2 | 69.5 | vs hard F0=`+0.1/-0.5/+0.3/+0.1`；bootstrap端点；AMP skip累计1 |
| 4090 | MR-F0 | 20 | 46.5 | 57.6 | 71.7 | 76.5 | vs hard F0=`+0.2/+0.3/-0.4/-0.3`；anchor `26/26`有限变化，adapter `0/6` changed |
| 4090 | MR-F0 | 30 | 51.6 | 62.9 | 75.9 | 80.2 | vs hard F0=`-0.6/-0.6/+0.2/-0.3`；`<e60`不作负裁决 |
| 4090 | MR-F0 | 40 | 52.9 | 64.9 | 77.7 | 81.8 | vs hard F0=`-0.1/+0.7/+1.1/+0.2`；继续 |
| 4090 | MR-F0 | 50 | 53.6 | 64.4 | 78.0 | 82.4 | vs hard F0=`-0.5/-1.1/-0.4/-0.6`；继续 |
| 4090 | MR-F0 | 60 | 55.2 | 66.9 | 78.9 | 83.3 | vs hard F0=`+0.0/-0.4/+0.0/+0.4`；继续完整运行 |
| 4090 | MR-F0 | 70 | 55.2 | 66.7 | 78.7 | 82.9 | vs hard F0=`-0.4/-0.9/-0.9/-1.1`；不提前停止 |
| 4090 | MR-F0 | 80 | 55.2 | 65.8 | 78.7 | 82.7 | vs hard F0=`-0.3/-0.6/-0.5/-0.3`；继续 |
| 4090 | MR-F0 | 90 | 56.0 | 67.3 | 79.3 | 83.7 | vs hard F0=`+0.2/+0.1/-0.2/+0.4`；等待后续eval |
| 4090 | MR-F0 | 100 | 55.9 | 66.8 | 79.1 | 83.3 | vs hard F0=`+0.2/-0.3/-0.1/+0.1`；等待e110/final |
| 4090 | MR-F0 | 110 | 56.0 | 67.1 | 79.1 | 83.3 | vs hard F0=`+0.1/-0.4/-0.3/-0.1`；等待final |
| 4090 | MR-F0 | 120 | 56.0 | 67.1 | 79.2 | 83.4 | vs hard F0=`+0.1/-0.3/-0.1/+0.1`；vs B0=`+0.9/+0.4/-0.3/-0.4`；final有效 |
| 4090 | MR-P0 | 10 | 36.4 | 46.4 | 61.1 | 67.4 | vs hard P0=`-0.8/-0.2/-1.4/-1.5`；bootstrap端点 |
| 4090 | MR-P0 | 20 | 47.0 | 58.5 | 71.9 | 77.1 | vs hard P0=`+0.5/+1.4/+1.7/+0.9`；anchor/adapter均有限更新 |
| 4090 | MR-P0 | 30 | 53.1 | 64.9 | 76.9 | 81.8 | vs hard P0=`+1.5/+1.6/+2.4/+2.1`；继续到final |
| 4090 | MR-P0 | 40 | 53.4 | 65.2 | 78.0 | 82.4 | vs hard P0=`+0.7/+1.3/+2.0/+1.8`；继续 |
| 4090 | MR-P0 | 50 | 53.9 | 65.8 | 78.3 | 82.6 | vs hard P0=`+0.2/+0.7/+1.5/+1.3`；继续等待收敛 |
| 4090 | MR-P0 | 60 | 55.2 | 66.7 | 78.3 | 82.4 | vs hard P0=`+0.6/+0.9/+0.2/+0.0`；继续 |
| 4090 | MR-P0 | 70 | 55.2 | 67.0 | 78.6 | 83.0 | vs hard P0=`+0.2/+0.9/+0.2/+0.1`；继续 |
| 4090 | MR-P0 | 80 | 55.3 | 66.7 | 78.8 | 82.8 | vs hard P0=`+0.4/+0.8/+1.2/+0.3`；继续等待final |
| 4090 | MR-P0 | 90 | 55.9 | 67.2 | 79.4 | 83.5 | vs hard P0=`+0.2/-0.2/+0.6/+0.1`；继续 |
| 4090 | MR-P0 | 100 | 55.6 | 67.1 | 78.8 | 82.7 | vs hard P0=`+0.1/+0.4/+0.6/-0.4`；继续 |
| 4090 | MR-P0 | 110 | 55.7 | 67.2 | 78.7 | 82.7 | vs hard P0=`+0.2/+0.5/+0.4/-0.3`；等待final |
| 4090 | MR-P0 | 120 | 55.7 | 67.1 | 78.7 | 82.9 | vs hard P0=`+0.1/+0.4/+0.3/-0.1`；vs B0=`+0.6/+0.4/-0.8/-0.9`；final有效 |
| 4090 | D0-same | 10 | 36.1 | 45.0 | 60.3 | 66.7 | vs hard F0=`-1.7/-2.6/-2.6/-2.7`；bootstrap端点，持续pose监督 |
| 4090 | D0-same | 20 | 45.3 | 55.6 | 69.8 | 75.3 | vs hard F0=`-1.0/-1.7/-2.3/-1.5`；anchor更新、adapter逐位不变 |
| 4090 | D0-same | 30 | 52.2 | 64.0 | 76.3 | 80.5 | vs hard F0=`+0.0/+0.5/+0.6/+0.0`；持续pose监督 |
| 4090 | D0-same | 40 | 53.2 | 63.9 | 77.6 | 81.9 | vs hard F0=`+0.2/-0.3/+1.0/+0.3`；继续到e60/final |
| 4090 | D0-same | 50 | 53.4 | 64.5 | 78.0 | 81.9 | vs hard F0=`-0.7/-1.0/-0.4/-1.1`；持续pose监督 |
| 4090 | D0-same | 60 | 55.6 | 67.1 | 79.3 | 82.7 | vs hard F0=`+0.4/-0.2/+0.4/-0.2`；继续到final |
| 4090 | D0-same | 70 | 55.4 | 66.2 | 79.5 | 83.5 | vs hard F0=`-0.2/-1.4/-0.1/-0.5`；matched轨迹混合 |
| 4090 | D0-same | 80 | 55.5 | 66.1 | 79.0 | 82.7 | vs hard F0=`+0.0/-0.3/-0.2/-0.3`；继续 |
| 4090 | D0-same | 90 | 56.3 | 67.6 | 79.8 | 83.5 | vs hard F0=`+0.5/+0.4/+0.3/+0.2`；四项全正 |
| 4090 | D0-same | 100 | 56.1 | 67.4 | 80.0 | 83.3 | vs hard F0=`+0.4/+0.3/+0.8/+0.1`；等待final |
| 4090 | D0-same | 110 | 56.1 | 67.5 | 79.7 | 83.3 | vs hard F0=`+0.2/+0.0/+0.3/-0.1`；等待final |
| 4090 | D0-same | 120 | 56.2 | 67.6 | 79.8 | 83.4 | final；vs hard F0=`+0.3/+0.2/+0.5/+0.1`；vs B0=`+1.1/+0.9/+0.3/-0.4` |
| 4090 | J0-same | 10 | 36.5 | 46.5 | 61.9 | 68.4 | vs fresh D0=`+0.4/+1.5/+1.6/+1.7`；bootstrap端点，不作裁决 |
| 4090 | J0-same | 20 | 46.1 | 57.2 | 70.4 | 75.1 | vs fresh D0=`+0.8/+1.6/+0.6/-0.2`；anchor/adapter均有限更新 |
| 4090 | J0-same | 30 | 52.7 | 63.8 | 77.0 | 81.8 | vs fresh D0=`+0.5/-0.2/+0.7/+1.3`；继续到final |
| 4090 | J0-same | 40 | 53.8 | 65.1 | 78.5 | 82.8 | vs fresh D0=`+0.6/+1.2/+0.9/+0.9`；早中期四项全正 |
| 4090 | J0-same | 50 | 52.3 | 64.0 | 76.7 | 81.3 | vs fresh D0=`-1.1/-0.5/-1.3/-0.6`；matched轨迹反转 |
| 4090 | J0-same | 60 | 55.8 | 67.6 | 78.9 | 83.1 | vs fresh D0=`+0.2/+0.5/-0.4/+0.4`；继续到final |
| 4090 | J0-same | 70 | 55.4 | 67.0 | 78.7 | 83.1 | vs fresh D0=`+0.0/+0.8/-0.8/-0.4`；matched轨迹仍混合 |
| 4090 | J0-same | 80 | 55.8 | 66.7 | 79.2 | 83.3 | vs fresh D0=`+0.3/+0.6/+0.2/+0.6`；四项小正 |
| 4090 | J0-same | 90 | 56.4 | 68.1 | 79.9 | 83.8 | vs fresh D0=`+0.1/+0.5/+0.1/+0.3`；等待final |
| 4090 | J0-same | 100 | 56.1 | 67.7 | 79.3 | 83.8 | vs fresh D0=`+0.0/+0.3/-0.7/+0.5`；等待final |
| 4090 | J0-same | 110 | 56.2 | 68.0 | 79.5 | 83.8 | vs fresh D0=`+0.1/+0.5/-0.2/+0.5`；等待final |
| 4090 | J0-same | 120 | 56.2 | 67.9 | 79.5 | 83.9 | final；vs fresh D0=`+0.0/+0.3/-0.3/+0.5`；vs B0=`+1.1/+1.2/+0.0/+0.1` |
| 4090 | R0 | 10 | 37.7 | 47.6 | 62.2 | 68.6 | vs B0=`+0.8/+0.6/+0.4/+0.3`；fixed external target PSG |
| 4090 | R0 | 20 | 46.7 | 56.5 | 71.0 | 76.8 | vs B0=`+4.4/+2.8/+5.3/+6.0`；`<e60`不作裁决 |
| 4090 | R0 | 30 | 52.4 | 63.8 | 75.9 | 81.0 | vs B0=`+1.8/+1.9/+0.7/+0.6`；继续到final |
| 4090 | R0 | 40 | 54.1 | 65.3 | 78.6 | 82.4 | vs B0=`+1.1/+0.2/+1.0/+0.5`；继续到final |
| 4090 | R0 | 50 | 53.3 | 64.8 | 77.5 | 82.7 | vs B0=`+1.2/+1.3/+0.6/+1.7`；继续到final |
| 4090 | R0 | 60 | 54.9 | 66.2 | 78.2 | 82.5 | vs B0=`+1.1/+1.0/+0.5/+0.5`；继续到final |
| 4090 | R0 | 70 | 55.5 | 66.8 | 79.0 | 83.4 | vs B0=`+1.1/+0.4/+0.1/+0.3`；等待final |
| 4090 | R0 | 80 | 55.5 | 66.6 | 79.0 | 83.3 | vs B0=`+0.9/+0.0/+0.1/+0.2`；等待final |
| 4090 | R0 | 90 | 56.0 | 67.4 | 79.6 | 84.1 | vs B0=`+1.1/+1.0/+0.2/+0.9`；等待final |
| 4090 | R0 | 100 | 56.0 | 66.7 | 79.3 | 83.7 | vs B0=`+1.2/+0.1/+0.0/+0.4`；等待final |
| 4090 | R0 | 110 | 56.1 | 67.4 | 79.4 | 83.8 | vs B0=`+1.0/+0.6/-0.1/+0.2`；等待final |
| 4090 | R0 | 120 | 56.1 | 67.4 | 79.5 | 83.7 | final；vs B0=`+1.0/+0.7/+0.0/-0.1`；vs D0=`-0.1/-0.2/-0.3/+0.3` |
| 4090 | RG0 | 10 | 38.6 | 49.3 | 64.6 | 71.0 | vs matched R0=`+0.9/+1.7/+2.4/+2.4`；`<e60`不裁决 |
| 4090 | RG0 | 20 | 47.1 | 57.9 | 71.9 | 77.6 | vs matched R0=`+0.4/+1.4/+0.9/+0.8`；PSG `8/8`有限更新 |
| 4090 | RG0 | 30 | 52.2 | 63.9 | 76.6 | 81.5 | vs matched R0=`-0.2/+0.1/+0.7/+0.5`；继续到final |
| 4090 | RG0 | 40 | 54.1 | 65.7 | 77.9 | 82.8 | vs matched R0=`+0.0/+0.4/-0.7/+0.4`；继续 |
| 4090 | RG0 | 50 | 54.4 | 66.0 | 78.7 | 83.1 | vs matched R0=`+1.1/+1.2/+1.2/+0.4`；四项全正 |
| 4090 | RG0 | 60 | 55.4 | 66.9 | 79.1 | 83.4 | vs matched R0=`+0.5/+0.7/+0.9/+0.9`；继续到final |
| 4090 | RG0 | 70 | 55.9 | 67.0 | 80.2 | 84.0 | vs matched R0=`+0.4/+0.2/+1.2/+0.6`；四项全正 |
| 4090 | RG0 | 80 | 56.0 | 67.4 | 79.8 | 84.0 | vs matched R0=`+0.5/+0.8/+0.8/+0.7`；四项全正 |
| 4090 | RG0 | 90 | 56.3 | 67.3 | 80.1 | 83.9 | vs matched R0=`+0.3/-0.1/+0.5/-0.2`；继续到final |
| 4090 | RG0 | 100 | 56.2 | 67.1 | 79.8 | 84.0 | vs matched R0=`+0.2/+0.4/+0.5/+0.3`；等待final |
| 4090 | RG0 | 110 | 56.2 | 67.0 | 80.0 | 83.8 | vs matched R0=`+0.1/-0.4/+0.6/+0.0`；等待final |
| 4090 | RG0 | 120 | 56.2 | 66.9 | 79.8 | 83.9 | final；vs R0=`+0.1/-0.5/+0.3/+0.2`；vs B0=`+1.1/+0.2/+0.3/+0.1` |
| 4090 | N0 | 10 | 37.3 | 46.7 | 62.6 | 68.5 | vs matched hard F0=`-0.5/-0.9/-0.3/-0.9`；bootstrap端点，`<e60`不裁决 |
| 4090 | N0 | 20 | 46.6 | 57.5 | 71.5 | 77.0 | vs matched hard F0=`+0.3/+0.2/-0.6/+0.2`；anchor/adapter逐位不变 |
| 4090 | N0 | 30 | 52.9 | 64.8 | 77.0 | 81.4 | vs matched hard F0=`+0.7/+1.3/+1.3/+0.9`；`<e60`不裁决 |
| 4090 | N0 | 40 | 54.0 | 66.1 | 78.4 | 82.0 | vs matched hard F0=`+1.0/+1.9/+1.8/+0.4`；继续到final |
| 4090 | N0 | 50 | 53.9 | 66.0 | 78.6 | 82.9 | vs matched hard F0=`-0.2/+0.5/+0.2/-0.1`；继续 |
| 4090 | N0 | 60 | 55.4 | 67.8 | 79.1 | 83.2 | vs matched hard F0=`+0.2/+0.5/+0.2/+0.3`；不以单点定案 |
| 4090 | N0 | 70 | 55.5 | 66.7 | 78.6 | 83.1 | vs matched hard F0=`-0.1/-0.9/-1.0/-0.9`；matched轨迹反转 |
| 4090 | N0 | 80 | 55.6 | 66.9 | 79.2 | 83.3 | vs matched hard F0=`+0.1/+0.5/+0.0/+0.3`；继续 |
| 4090 | N0 | 90 | 56.2 | 67.9 | 79.9 | 83.3 | vs matched hard F0=`+0.4/+0.7/+0.4/+0.0`；等待后续eval |
| 4090 | N0 | 100 | 56.0 | 67.5 | 79.6 | 83.3 | vs matched hard F0=`+0.3/+0.4/+0.4/+0.1`；等待final |
| 4090 | N0 | 110 | 56.1 | 67.6 | 80.0 | 83.5 | vs matched hard F0=`+0.2/+0.1/+0.6/+0.1`；等待final |
| 4090 | N0 | 120 | 56.1 | 67.6 | 80.0 | 83.4 | final；vs hard F0=`+0.2/+0.2/+0.7/+0.1`；vs B0=`+1.0/+0.9/+0.5/-0.4` |
