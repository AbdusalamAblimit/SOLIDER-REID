# exp370 PBSR 监控

## 状态

- 当前阶段：第一批 B0/P0 已按冻结 manifest 并行训练
- 训练状态：B0 进行中；P0 进行中
- exact execution commit：`14b2b68`
- source archive SHA256：`188df5e7049ed5b9bc877bdbe7965e5853e4e653711da09496707d16970664ee`
- B0 输出：3090 `/root/work/SOLIDER-REID-exp370-14b2b68/log/occluded_duke/exp370_b0_global_s1234`
- P0 输出：4090 `/home/afr/SOLIDER-REID-exp370-14b2b68/log/occluded_duke/exp370_p0_coupled_s1234`

## 预训练门禁

- [x] 核对 PAFormer 与 LGPA 的直接重合
- [x] 核对 ProFD reverse cross-attention 的真实数据流
- [x] 核对 KPR、BPBreID、PFD、PAT、TSD、PGDS/PGFL-KD
- [x] 写明可主张与不可主张的新颖性边界
- [x] 纳入 exp161、exp320 等历史负证据
- [x] 完成代码实现前独立 design 审查
- [x] 完成默认关闭与零初始化逐元素退化测试
- [x] 完成 pose-loss/backbone 梯度防火墙测试
- [x] 完成 eval 无姿态依赖测试
- [x] 完成 CPU bfloat16 autocast smoke test
- [x] 完成远程真实 dataloader + CUDA AMP smoke test
- [x] 冻结 kill-switch manifest 后方可启动训练

## 事件记录

### [2026-07-13] ProFD 风险裁决

- 官方实现的 reverse cross-attention 会更新 visual token 副本，但该副本只作为 part decoder 内部 key/value。
- 最终 global descriptor 仍取原始 CLIP CLS；更新后的 visual tokens 不返回主干。
- 结论：不能声称“首次双向 attention”，但“共享路由重组实际 global 主表征”仍保有差异空间。

### [2026-07-13] 机制收紧

- 删除 CLIP text prototype 依赖。
- pose 从前向 bias 改成纯训练监督 target。
- read/write 强制共享 routing matrix。
- pose assignment loss 使用 detached backbone 输入。
- 最终只返回标准 global descriptor，不使用 part concat/MaxSim。
- 当前判断：允许进入无训练实现与审计，不允许直接开正式训练。

### [2026-07-13] 本地实现审计 PASS

- 新增独立模块 `model/modules/pose_bidirectional_router.py`，所有 config 默认关闭。
- read/write 共享 routing；independent-write 对照不增加参数。
- route loss 用 `feat_map.detach()` 重新计算路由，router 有梯度、input/backbone 无梯度。
- `write_scale=0` 时输出与输入 bitwise 相同；首步 `write_scale.grad` 非零。
- 门打开后 identity probe 可到达 key/query/out projection。
- eval 对 None/correct/random heatmap bitwise 不变。
- coupled 与 independent 参数量相同。
- CPU bfloat16 autocast forward/backward finite。
- YACS 成功读取冻结 `configs/occluded_duke/exp370_pbsr.yml`，batch size 保持 64。
- 本地测试输出：`PBSR mechanism checks: PASS`。
- 当前判断：允许远程单批次 CUDA smoke，仍不允许直接启动 120 epoch 正式训练。

### [2026-07-13] 3090 真实 CUDA smoke PASS

- 隔离目录：`/root/work/SOLIDER-REID-exp370`，未修改原 dirty repo。
- 真实 Occluded-Duke dataloader、Swin-Tiny 预训练权重、batch size 64、标准 ID/triplet、route loss、CUDA AMP、生产 optimizer 全链路执行一批。
- 历史默认 AMP scale `65536` 在 P0 和纯 B0 上均产生首批 backbone overflow，排除 PBSR 特有错误；P0 在 `2048` 通过，矩阵统一保守冻结为 `1024`。
- P0：identity `21.99402428`，route `1.52670991`，total `22.75737953`；203 个有梯度参数全部 finite，177 个 nonzero。
- write-scale gradient `1.87530518e-02`，optimizer step 后 `0 -> 1.50024407e-05`；slot query、key projection、backbone gradient 均 finite/nonzero。
- 初始化 route entropy `3.87105513`、background share `0.14284959` 与均匀初始化预期一致；delta norm `2.41153574` finite。
- 同 scale 的 B0：identity `21.99402428`，173/173 个有梯度参数 finite/nonzero；零门使 P0/B0 初始 identity 前向严格一致。
- 当前判断：smoke 门禁通过，manifest FROZEN，允许只启动第一批 B0/P0；尚无训练结果，不得声称 PBSR 有效。

### [2026-07-13] B0/P0 第一批启动

- B0 于 3090 启动，唯一训练 main PID `1698212`；其余同命令进程为 DataLoader workers，不是第二 controller。
- P0 于 4090 启动，唯一训练 main PID `4118839`；其余同命令进程为 8 个 DataLoader workers。
- 两边均从 exact archive 解包到新隔离目录，未修改原 dirty repo；batch size 均为 64，AMP initial scale 均为 1024。
- 两边 torch/torchvision 对齐为 `2.4.1+cu121 / 0.19.1+cu121`。4090 使用工作目录内 `uv` 创建的 `.venv` 入口，并显式组合只读依赖路径；其 CUDA smoke 已再次通过。
- B0 epoch 10：`36.0% mAP / 44.8% R1`；epoch 20：`43.3% mAP / 53.5% R1`，来自原始 `runner_stdout.log`。
- P0 epoch 5 完成：训练 loss finite；route loss 从首个日志点约 `1.556` 缓降至约 `1.547`；route entropy `3.871`、background share `0.143`、delta norm 约 `2.32`，无 NaN/Inf 或 background collapse。
- P0 的 alpha 日志以三位小数显示，早期接近 0 不能据此判断未更新；CUDA smoke 已验证 optimizer step 后非零。后续以 checkpoint 参数和更高精度审计为准。
- 当前判断：两臂健康继续。P0 尚无 eval，不能与 B0 作效果判断；等待相同 epoch 10/20/30 门禁。

### [2026-07-13] epoch 30 kill-switch：继续到 epoch 60

| Epoch | B0 mAP / R1 | P0 mAP / R1 | P0-B0 mAP / R1 |
|---:|---:|---:|---:|
| 10 | 36.0 / 44.8 | 34.2 / 43.0 | -1.8 / -1.8 |
| 20 | 43.3 / 53.5 | 38.4 / 48.1 | -4.9 / -5.4 |
| 30 | 48.9 / 58.6 | 48.5 / 57.8 | -0.4 / -0.8 |

- 所有数值均从各自隔离目录的原始 `runner_stdout.log` 读取。
- P0 在 epoch 20 出现明显落后，但 epoch 30 恢复到预注册的 `±1.0 mAP` 继续区间，未触发“落后超过 1.0 mAP”早停。
- epoch 30 P0 route loss 约 `1.13`，相对初始约 `1.56` 明显下降；alpha 约 `-0.029`，route entropy 从 `3.871` 降至约 `3.718`，background share 约 `0.158`，delta norm 约 `2.21`。
- 机制统计显示 router 与写回门正在学习，无 NaN/Inf、dead route 或 background collapse；alpha 为负只表示残差方向由 identity loss 学得，不构成异常。
- 当前判断：严格按 manifest 继续到 epoch 60；当前 P0 仍未超过 B0，不启动 P1/P4，也不补任何救场变体。

### [2026-07-13] epoch 60 跨机 screening：负向，终止首轮 B0/P0

| Epoch | B0 mAP / R1 | P0 mAP / R1 | P0-B0 mAP / R1 |
|---:|---:|---:|---:|
| 40 | 51.8 / 61.7 | 51.4 / 61.0 | -0.4 / -0.7 |
| 50 | 52.4 / 62.6 | 53.2 / 63.5 | +0.8 / +0.9 |
| **60（冻结门禁）** | **55.3 / 64.7** | **54.4 / 63.7** | **-0.9 / -1.0** |

- epoch 60 是 manifest 在 epoch 30 未触发强早停后的第二裁决点；P0 不仅未达到目标 `+0.8～1.0 mAP`，而且相对 B0 为负，因此不具备“明确正向”。
- epoch 50 的单点 `+0.8/+0.9` 在 epoch 60 反转，说明当前证据不稳定，不能挑最好中间点宣称有效。
- 等待较慢 B0 的 epoch 60 时，P0 自动运行到 epoch 80：epoch 70 为 `56.9/66.6`，epoch 80 为 `57.9/67.6`。这两个值缺少同 epoch B0，只作为训练完整性诊断，**不用于因果比较**。
- epoch 60 机制统计仍健康：route loss 约 `0.90`、alpha 约 `-0.042`、entropy 约 `3.36`、background share 约 `0.196`、delta norm 约 `2.06`。因此失败不是 NaN、死门或 background collapse，而是“路由确实学会了 pose target，但写回没有改善 identity global”。
- 首轮执行已精确终止：B0 在 epoch 64 训练中、P0 在 epoch 85 训练中收到 TERM；两张 GPU 已释放。B0 保留 epoch 20/40/60 checkpoint，P0 保留 20/40/60/80 checkpoint；原始日志已回传到 `execution_14b2b68/`。
- **执行完整性复核修正**：B0 与 P0 分别运行在 3090/4090；torch/torchvision 虽已对齐，但 Python、NumPy/Pillow/timm 等没有逐项完全相同。epoch 60 的 `-0.9 mAP` 不足以压过这个非方法混杂，故此处只能记为“跨机 screening 负向”，不能冒充最终严格 NO-GO。
- 最终裁决前只补一个必要的同机控制：在 4090、使用 P0 完全相同解释器、依赖路径、GPU、源码和 config 重跑 `POSE_PBSR=False` 的 B0 到 epoch 60。该控制不改变方法、不启动 P1/P4/P2/P3、不做超参救场。
- 若同机 B0 仍高于或接近 P0、P0 未达到明确 `+0.8～1.0 mAP`，则正式 NO-GO；只有同机结果证明 P0 明确正向，才允许重新进入机制消融门禁。

### [2026-07-13] 4090 同运行时 B0 控制启动

- 唯一目的：消除首轮 B0/P0 的跨机器与依赖版本混杂；这不是新方法变体，也不改变预注册裁决点。
- 执行目录：`/home/afr/SOLIDER-REID-exp370-14b2b68`；关键 config、PBSR 模块和 processor 的 SHA256 均与 execution commit `14b2b68` 对应文件一致。
- 运行时与 P0 相同：Python `3.10.12`，torch/torchvision `2.4.1+cu121 / 0.19.1+cu121`，NumPy `2.2.6`，Pillow `12.2.0`，timm `1.0.27`。
- 输出目录：`log/occluded_duke/exp370_b0_global_sameenv_s1234`；唯一 main PID `4143251`，其 8 个子进程为 DataLoader workers。
- 命令行唯一方法覆盖为 `MODEL.POSE_PBSR False`；日志确认 `POSE_PBSR=False`、`MAX_EPOCHS=120`、seed `1234`、batch size `64`、AMP initial scale `1024`。
- epoch 1 已完成：loss finite，`Loss=13.083`、`id_global=6.554`、`tri_global=6.530`；GPU 仅有 main PID 占用约 `6.95 GiB`，无重复 controller、NaN/Inf 或异常退出。
- 当前判断：健康继续。保持 120-epoch cosine schedule，在 epoch 60 eval 落盘后精确终止；最终只与现有同机 P0 epoch 60 的 `54.4 mAP / 63.7 R1` 比较，不挑选其他 epoch。

### [2026-07-13] epoch 60 同机同运行时最终门禁：PBSR P0 正式 NO-GO

| Epoch | B0 same-env mAP / R1 | P0 mAP / R1 | P0-B0 mAP / R1 |
|---:|---:|---:|---:|
| 10 | 33.4 / 42.7 | 34.2 / 43.0 | +0.8 / +0.3 |
| 20 | 43.1 / 53.3 | 38.4 / 48.1 | -4.7 / -5.2 |
| 30 | 49.2 / 59.5 | 48.5 / 57.8 | -0.7 / -1.7 |
| 40 | 52.8 / 62.5 | 51.4 / 61.0 | -1.4 / -1.5 |
| 50 | 53.0 / 63.3 | 53.2 / 63.5 | +0.2 / +0.2 |
| **60（冻结门禁）** | **54.5 / 63.8** | **54.4 / 63.7** | **-0.1 / -0.1** |

- 同机 B0 使用 P0 完全相同的 4090、Python、依赖路径、execution 源码、config、seed、batch size 与 AMP scale；唯一方法差异是 `POSE_PBSR=False`。跨机器/runtime 混杂已消除。
- 所有 eval 均从原始日志逐项列出。P0 只在 epoch 10/50 出现 `+0.8/+0.2 mAP` 的孤立领先，epoch 20/30/40 为负，冻结 epoch 60 为 `-0.1/-0.1`；禁止挑选中间点。
- 只读 gate watcher 在 epoch 60 验证完整写到 Rank-10 后发送 TERM；训练刚进入 epoch 61 iter 50，随后 main/workers 退出，GPU 释放。全程无 Traceback、NaN/Inf、OOM、RuntimeError 或 ERROR。
- P0 的 route loss、alpha、entropy、background share 与 delta norm 均表明机制确实学习且没有结构性崩溃，但这种 pose-supervised route learning 没有转化为 global identity retrieval 增益。
- **正式裁决：P0 未达到 `+0.8～1.0 mAP` 门槛，PBSR NO-GO。** 依据预注册顺序停止 P1/P4/P2/P3、三 seed和跨骨干；不做超参救场或 PBSR 小变体。
- 结论边界：这否定的是当前“共享路由读取—槽间混合—同路由写回 global”机制作为论文主贡献；不否定历史 LGPA/pose 分支信号，也不能声称 pose 普遍无效。因为主门禁已失败，uniform/shuffled 因果消融不再启动，故不能声称正确 pose 优于这些控制。
- 原始同机日志已回传到 `execution_14b2b68/`，SHA256 与远端一致；完整证据见该目录 `manifest.md`。
