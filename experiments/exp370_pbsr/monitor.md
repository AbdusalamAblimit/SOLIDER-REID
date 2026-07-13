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
