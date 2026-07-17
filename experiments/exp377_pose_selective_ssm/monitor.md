# exp377 Pose-Conditioned Selective SSM 运行监控

## 当前状态

- 状态：`COMPLETE_NO_GO`
- 前序：exp376 已在 e60 性能 Gate 判为 NO-GO 并停止；两机 GPU 已清空。
- 机制：最终 `12×4` RGB token 上的纯 PyTorch 双向 selective SSM；实例姿态联合修正
  `Δ/B/C`，固定 serpentine scan。
- 已完成首轮：4090 P0（instance pose），3090 D0（同参数 RGB-only）均自然跑满 e120。
- 判定纪律：跨机只作趋势；正式 NO-GO 由 P0 e60 相对预注册同机 clean B0 的性能门禁触发。

## 启动前阻塞项

- [x] standalone 模块测试：stable A、serpentine/reverse、zero=D0、correct≠zero、梯度、reload；
- [x] production Swin 集成与 target-only heatmap 审计；
- [x] CPU smoke 与两机 forward/backward；
- [x] batch64 CUDA AMP + GradScaler：finite、无 underflow、关键参数实际更新；
- [x] P0/D0/M0 初始化与 shared SSM state dict 精确一致；
- [x] e60 冻结反事实 evaluator：8 arms、frozen donor、final-grid support/composition
  exact audit、pose-off 与 correct-start/end；本地/4090 utility tests `4/4 PASS`；
- [x] Codex 最终实现审查通过：见 `codex_review.md`；
- [x] 独立 OUTPUT_DIR、唯一 main 与启动后异常扫描确认。

## 实现与 preflight 结果

- 独立模块测试：本地、3090、4090 均 `6/6 PASS`；覆盖 serpentine 互逆、stable `A`、
  true `exp(ΔA)` recurrence、pose 对 `Δ/B/C` 的联合控制、zero=D0、visibility/composition
  解耦、CPU bfloat16 与全分支梯度。
- 4090 production-model integration：`PASS`。P0/D0/M0 state dict 逐键相同；模块实际收到
  person-0 target heatmap 而非 scene max-merge；P0、D0、M0 descriptor 均 finite 且按预期不同；
  strict reload 精确复现。exp377 B0 与 legacy exp375 B0 的 state dict、descriptor 和最终 featmap
  均逐元素精确相同，默认路径未被破坏。
- 纯 PyTorch 模块参数量：`328,996`；两机均不依赖 `mamba_ssm`、Triton 或自编译 CUDA 扩展。

### 4090 P0 batch64 AMP

- loss `0.598913`，峰值显存 `5.651 GiB / 23.643 GiB`；
- `Δ min/mean/p95/max = 0.001000 / 0.011742 / 0.025461 / 0.099777`；
- `A=[-16,-1]`，`dA mean=0.908322`，state RMS/max=`0.048462/1.21894`；
- pose `Δ/B/C` residual RMS=`0.001591/0.001643/0.001668`；
- output delta ratio=`0.003058`；全部 finite；
- `A/Δ/B/C`、pose MLP、三个 pose gain、output projection、alpha 均有 finite 非零梯度，
  optimizer step 后全部实际更新。

### 3090 D0 batch64 AMP

- loss `0.596020`，峰值显存 `5.672 GiB / 23.690 GiB`；
- `Δ min/mean/p95/max = 0.001000 / 0.011734 / 0.025317 / 0.100000`；
- `A=[-16,-1]`，`dA mean=0.908344`，state RMS/max=`0.048525/1.19256`；
- pose `Δ/B/C` residual RMS 精确为 `0/0/0`，RGB selective core 与 output/alpha 均有有限
  非零梯度并实际更新；
- output delta ratio=`0.003180`；全部 finite。

P0 与 D0 的数值只证明各自在目标运行时健康，不能把跨机 loss/统计差当作实验差值。

## e60 冻结反事实执行准备

`eval_counterfactual.py` 已能在同一 P0 checkpoint/model instance 下评测 correct-start/end、
frozen matched pose、recipient visibility + donor composition、donor visibility + recipient
composition、joint-channel permutation、canonical 与 pose-off，并强制 RGB/path/PID/camera 顺序
一致。support/composition 交换先使用模块相同的 bilinear 操作降到最终 `12×4`，再直接构造
final-grid 输入；evaluator 调用模块自身 `_local_pose` 复核 active composition 与 visibility，
任一 max error `>1e-5` 即失败。source-empty 位置使用同一 donor 的全局 joint composition 作为
显式 fallback。

exp377-specific matched donor 的最终 nuisance preflight 必须等 e60 checkpoint 形成后，按当时
实际 pose MLP/gain 与 `Δ/B/C` residual/Δ 分布生成；不得直接把 exp375 donor map 的 PASS 继承为
本实验 PASS。该步骤是 e60 裁决前置项，不是正式训练启动前可伪造的静态检查。

最终未执行 donor mapping 与冻结反事实：P0 e60 已先以 `54.5 mAP` 相对预注册同机 clean
B0 的 `55.2 mAP` 低 `0.7`，满足“低至少 `0.5 mAP` 即 NO-GO”的充分条件。性能门禁已经失败，
继续构造 matched donor 不会改变停止决定，也不允许用未经执行的反事实声称 pose 因果结论。

## 正式执行启动

- exact execution commit：`52c4ef6dc93b2afc5439b96d59dc044e7b448fd5`；
- 4090 clean detached worktree：`/home/afr/SOLIDER-REID-exp377-52c4ef6`；
- 3090 clean detached worktree：`/root/work/SOLIDER-REID-exp377-52c4ef6`；
- 4090 P0 OUTPUT：`log/occluded_duke/exp377_p0_pose_ssm_s1234`，main PID `167138`；
- 3090 D0 OUTPUT：`log/occluded_duke/exp377_d0_rgb_ssm_s1234`，main PID `1776864`；
- 两边均确认一个 main + 8 DataLoader workers、独立空 OUTPUT_DIR、正确 pretrained/data、
  GPU compute 活跃；未启动 controller 或第二训练。

4090 P0 epoch1 iter100 已健康：loss `14.995`，alpha `9.983e-3`，gain `0.100`，
`Δ mean[p05,p95]=1.137e-2[3.946e-3,2.388e-2]`，`dA mean=0.9108`，pose `Δ/B/C`
residual RMS=`1.567e-3/1.492e-3/1.586e-3`，state RMS `3.142e-2`，output delta ratio
`4.360e-4`。启动日志确认 `source=input`、target-person heatmap、328,996 参数；未见
NaN/Inf/Traceback/RuntimeError/OOM。

3090 D0 epoch1 iter150 同样健康：loss `13.790`，alpha `9.965e-3`，gain `0.100`，
`Δ mean[p05,p95]=1.138e-2[3.949e-3,2.389e-2]`，`dA mean=0.9108`，pose `Δ/B/C`
residual RMS 精确为 `0/0/0`，state RMS `3.235e-2`，output delta ratio `4.430e-4`。
启动日志确认 `source=zero`；一个 main + 8 workers，未见异常。

### 2026-07-15 17:30 启动后健康检查

- 4090 P0：唯一 main PID `167138` + 8 workers，GPU `83%`、显存 `7.25/24.56 GB`；
  已完成 epoch 7，epoch 8 iter 100。最新 loss `6.343`，alpha `2.184e-2`，gain
  `9.998e-2`，`Δ mean[p05,p95]=1.147e-2[4.005e-3,2.476e-2]`，`dA=0.9102`，
  pose `Δ/B/C` residual RMS=`1.572e-3/1.489e-3/1.591e-3`，state RMS
  `3.011e-2`，output delta ratio=`1.300e-3`；各动态量 finite 且模块持续离开 identity。
- 3090 D0：唯一 main PID `1776864` + 8 workers，GPU `60%`、显存 `8.19/24.58 GB`；
  已完成 epoch 3，epoch 4 iter 100。最新 loss `7.498`，alpha `1.022e-2`，gain
  `0.100`，`Δ mean[p05,p95]=1.139e-2[3.806e-3,2.452e-2]`，`dA=0.9109`，
  pose `Δ/B/C` residual 仍精确为 `0/0/0`，state RMS `2.280e-2`，output delta
  ratio=`4.950e-4`；RGB selective SSM 正常学习。
- 两边 runner/train log 均未见 NaN、Inf 数值、Traceback、RuntimeError、OOM；尚未到 epoch 10，
  因此没有完整 eval，继续运行且不作性能判断。

### 2026-07-15 17:32 P0 epoch 10 eval

- 4090 P0 完成首次全量评估：`35.4 mAP / 45.6 R1 / 61.4 R5 / 67.0 R10`；
- epoch 10 末 alpha=`5.545e-2`，pose `Δ/B/C` residual 仍为有限非零值，output delta
  ratio=`4.901e-3`，模块已明显离开近 identity 初始化；
- 这是 `<e60` 的早期曲线点，只记录，不作正负裁决；训练已继续进入后续 epoch。

## 最终结果与裁决

两条训练均自然完成 e120，main/workers 已退出、GPU 空闲；runner/train log 未见 NaN、Inf、
Traceback、RuntimeError 或 OOM。P0 e120 的 alpha=`0.1808`、gain=`0.09916`，pose
`Δ/B/C` residual RMS 约 `1.51e-3/1.43e-3/1.52e-3`，state RMS 约 `5.43e-2`，
output delta ratio 约 `1.93e-2`。因此模块和 pose residual 都实际工作，失败不能归因于
identity 初始化、死分支或数值异常。

- 预注册 e60 clean B0：`55.2/65.0/77.6/83.1`；P0：`54.5/63.8/78.5/83.8`，
  P0−B0=`-0.7/-1.2/+0.9/+0.7`；mAP 已触发 `<=-0.5` 的 NO-GO 条件；
- e120 clean B0：`58.4/67.1/81.2/85.6`；P0：`58.6/67.8/81.4/86.3`，
  P0−B0=`+0.2/+0.7/+0.2/+0.7`，仍远低于正式 `+0.8 mAP` 门槛；
- 3090 D0 e120 为 `58.8/68.1/81.3/86.5`。跨机不能作正式差值，但 P0 没有优于
  RGB-only SSM 的趋势；
- 中间心跳没有在 e60 当场执行停止，故训练继续自然跑满；这额外提供了 e120 证据，但不改变
  预注册 e60 裁决。

**裁决：exp377 正式 NO-GO。** 不补同一 4090 B0/D0/M0，不运行 donor 反事实、多 seed、
跨 backbone 或 `Δ-only/B/C-only` 消融，也不把动态 scan、graph、更多 state、额外 loss 当作
临场救场。当前实现只说明普通 RGB-selective SSM 容量可以正常学习，不能支持“实例姿态校准
selective dynamics 带来身份检索收益”。

小型原始日志已回传至 `remote_artifacts/exp377_52c4ef6/`；远端 e60/e120 checkpoint 保留。

## 每次 eval 记录

| arm | epoch | mAP | R1 | R5 | R10 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| P0 | 10 | 35.4 | 45.6 | 61.4 | 67.0 | 继续；早期点，不裁决 |
| P0 | 20 | 41.1 | 50.8 | 65.4 | 71.5 | 继续；早期点，不裁决 |
| P0 | 30 | 49.8 | 59.3 | 74.0 | 79.6 | 继续；早期点，不裁决 |
| P0 | 40 | 50.7 | 59.8 | 74.6 | 80.6 | 继续；早期点，不裁决 |
| P0 | 50 | 53.4 | 62.2 | 77.9 | 83.7 | 继续至预注册门禁 |
| P0 | 60 | 54.5 | 63.8 | 78.5 | 83.8 | NO-GO：较 clean B0 低 0.7 mAP |
| P0 | 70 | 56.1 | 65.2 | 79.5 | 84.3 | 门禁后额外自然完成 |
| P0 | 80 | 57.6 | 66.6 | 80.8 | 85.9 | 门禁后额外自然完成 |
| P0 | 90 | 57.7 | 67.2 | 81.0 | 86.1 | 门禁后额外自然完成 |
| P0 | 100 | 58.6 | 67.9 | 81.6 | 86.5 | 门禁后额外自然完成 |
| P0 | 110 | 58.4 | 67.3 | 81.2 | 86.5 | 门禁后额外自然完成 |
| P0 | 120 | 58.6 | 67.8 | 81.4 | 86.3 | 完成；未达正式 mAP 门槛 |
| D0 | 10 | 34.0 | 42.7 | 59.0 | 65.5 | 跨机趋势 |
| D0 | 20 | 42.0 | 52.4 | 66.6 | 72.6 | 跨机趋势 |
| D0 | 30 | 49.8 | 59.5 | 74.7 | 80.9 | 跨机趋势 |
| D0 | 40 | 52.3 | 62.0 | 76.9 | 82.8 | 跨机趋势 |
| D0 | 50 | 53.4 | 63.8 | 78.0 | 82.7 | 跨机趋势 |
| D0 | 60 | 54.4 | 64.0 | 77.7 | 82.9 | 跨机趋势 |
| D0 | 70 | 56.6 | 65.8 | 80.4 | 85.2 | 跨机趋势 |
| D0 | 80 | 58.4 | 68.1 | 82.1 | 86.2 | 跨机趋势 |
| D0 | 90 | 58.0 | 67.7 | 81.6 | 86.5 | 跨机趋势 |
| D0 | 100 | 58.7 | 67.8 | 81.5 | 86.6 | 跨机趋势 |
| D0 | 110 | 58.8 | 67.9 | 81.5 | 86.6 | 跨机趋势 |
| D0 | 120 | 58.8 | 68.1 | 81.3 | 86.5 | 完成；无 P0 优势趋势 |

每次训练监控同时记录：唯一 main/8 workers、GPU、loss、alpha、A/dA、Δ p05/mean/p95/max、
RGB 与 pose `B/C` 范数、pose `Δ` residual、state RMS/max、output delta ratio，以及
NaN/Inf/Traceback/RuntimeError/OOM。
