# exp376 运行监控

## 当前状态

- 状态：`E60_GATE_NO_GO_STOPPED`
- 机制：stage 2/3 逐 block Pose Hyper-LoRA
- 计划并行：4090 P0（factor-wise basis）、3090 D0（exp071-style diagonal control）
- 本地机制测试：`5 passed`。
- 本地设计/代码两路 Codex 复审：`GO / GO`。
- 双机 exact execution commit：`57d145e3efcf95a2dddc15a58c006390419349bd`。

## 训练前 GPU 验证

两机 production-model integration 均 PASS：P0/M0 初始化逐键相同、8 层接入、target-person
热图、strict reload 与 descriptor 精确复现均通过。

batch64 CUDA AMP + GradScaler preflight：

| 机器/arm | factorization | 参数量 | changed fraction 范围 | applied RMS 范围 | 峰值显存 | 关键参数更新 |
|---|---|---:|---:|---:|---:|---|
| 4090 P0 | factor-wise basis | 137,288 | 60.9%–71.2% | 3.44e-6–7.39e-6 | 5.74 GiB | 8/8 A、B、pose MLP、α 全部更新 |
| 3090 D0 | diagonal | 139,400 | 84.1%–87.7% | 8.01e-6–1.18e-5 | 5.75 GiB | 8/8 A、B、pose MLP、α 全部更新 |

两路 loss finite，所有关键梯度 finite 且非零。原三维 bank 初始化造成的近 identity bug 已修复，
当前负结果将不再能归因于 FP16 舍入或模块未更新。

## 正式启动

- execution commit：`57d145e3efcf95a2dddc15a58c006390419349bd`
- 4090 P0 main PID：`142569`
- 3090 D0 main PID：`1767303`
- 两路均为唯一 main + 8 个 DataLoader workers，batch=64，seed=1234；GPU compute 正常。
- P0 epoch 1 已完成，31.5 秒/epoch；D0 已进入 epoch 1 后半段。
- P0 epoch 1 末运行统计：alpha mean `1.006e-3`，visibility `0.2121`，coefficient abs mean
  `0.02467`，raw delta RMS `0.004644`。
- D0 epoch 1 iter100：alpha mean `9.960e-4`，visibility `0.2131`，coefficient abs mean
  `0.02235`，raw delta RMS `0.009355`。
- 两路启动日志均确认 target-only heatmap、正确 factorization 与独立 OUTPUT_DIR；未见
  NaN / Inf / Traceback / RuntimeError / OOM。

## 监控字段

每次完整 eval 记录 epoch、mAP、R1、R5、R10；同时记录：

- 唯一 main PID 与 DataLoader workers；
- GPU 占用；
- `residual_scale`；
- pose coefficient abs mean；
- visibility mean；
- dynamic delta RMS；
- NaN / Inf / Traceback / OOM 扫描。

## Eval 轨迹

| arm | epoch | mAP | R1 | R5 | R10 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| P0 factor-wise | 10 | 35.1 | 44.2 | 59.6 | 66.1 | 健康；相对历史 B0 e10 +0.6 mAP / +0.9 R1 |
| D0 diagonal | 10 | 35.5 | 45.1 | 60.9 | 67.3 | 健康；早期高于 P0 0.4 mAP / 0.9 R1 |
| P0 factor-wise | 20 | 42.7 | 53.0 | 67.8 | 74.0 | 健康；与历史 B0 e20 42.8/53.3 基本重合 |
| P0 factor-wise | 30 | 49.8 | 59.8 | 74.7 | 80.2 | 健康；相对历史 B0 e30 -0.8 mAP / -0.7 R1，仍属早期轨迹 |
| P0 factor-wise | 40 | 51.2 | 60.5 | 75.8 | 81.6 | 相对历史 B0 e40 52.2/62.0 为 -1.0/-1.5 |
| D0 diagonal | 20 | 41.8 | 52.0 | 67.2 | 73.3 | 健康；跨机只记录趋势 |
| P0 factor-wise | 50 | 52.9 | 63.4 | 78.6 | 83.4 | 相对历史 B0 e50 53.6/63.7 为 -0.7/-0.3 |
| D0 diagonal | 30 | 48.4 | 58.1 | 72.5 | 79.0 | 同阶段低于 P0 1.4 mAP / 1.7 R1，但为跨机趋势 |
| P0 factor-wise | 60 | 54.2 | 63.0 | 76.8 | 82.1 | **中点性能 Gate NO-GO**；相对历史 B0 e60 55.2/65.0 为 -1.0/-2.0 |

两路 e10 均为唯一 main + 8 workers，GPU 正常；P0 已继续到 e22，D0 已继续到 e11。
异常扫描均为零。P0/D0 使用不同 GPU/运行时，当前差值只记录趋势，不作 factorization 裁决。

最新健康检查：P0 已进入 e35，D0 已完成 e18；两边均保持唯一 main + 8 workers，GPU、loss、
alpha/visibility/coefficient/delta 均为有限值，未见 NaN/Inf/Traceback/RuntimeError/OOM。

`< epoch 60` 只记录轨迹，不提前判负。

## 2026-07-16 — e60 中点裁决

P0 从 e30 起连续四个评测点不高于历史同机 clean B0：mAP 差依次为
`-0.8/-1.0/-0.7/-1.0`。e60 的 P0-B0 为 `-1.0 mAP / -2.0 R1`，距离预注册成功线
`P0-B0 >= +0.8 mAP` 仍差 1.8 mAP。模块并非未工作：全程 8 层 alpha、coefficient、
visibility、delta 均有限非零，batch64 AMP preflight 也已证明所有关键参数实际更新。

D0 e30 低于 P0 同阶段 1.4 mAP，但该差值来自不同 GPU/运行时，只能说明 factor-wise 版本
在当前筛查中不弱于 diagonal control，不能抵消 P0 对 clean baseline 的稳定负差。继续跑满即便
出现晚期小幅回升，也不太可能同时跨越 baseline 与 `+0.8` 成功门槛。

**裁决：exp376 在预注册 e60 性能 Gate 判为 NO-GO，停止 P0/D0，取消 M0、exact B0、
matched donor 与 visibility/coeff 反事实。** 结论边界只是否定当前 stage2/3 post-block、
rank4/M4 factor-wise Hyper-LoRA 实现；不外推否定所有 pose-conditioned low-rank operator。
下一机制按既定顺序进入真实 Mamba selective `Δ/B/C` update。

停止前先落盘上述裁决并提交，随后只向两个 main PID 发送 TERM；P0 于完整 e60 评测和
`transformer_60.pth` 落盘后停止（训练日志自然推进至 e63 完整训练段），D0 于完整 e30 评测后
停止（日志推进至 e32 iter100）。两个 main 与孤儿 workers 均已退出，GPU compute 清空。
未手工终止 worker、跳 epoch 或修改运行中代码/config。

本地保留的小型原始证据：

- `remote_artifacts/exp376/4090_p0/train_log.txt`
- `remote_artifacts/exp376/4090_p0/runner_stdout.log`
- `remote_artifacts/exp376/3090_d0/train_log.txt`
- `remote_artifacts/exp376/3090_d0/runner_stdout.log`

远端保留 e20/e40/e60 P0 checkpoint 与 e20 D0 checkpoint，未回传约 113 MB 的权重文件。
