# exp376 运行监控

## 当前状态

- 状态：`IMPLEMENTING`
- 机制：stage 2/3 逐 block Pose Hyper-LoRA
- 计划并行：4090 P0（factor-wise basis）、3090 D0（exp071-style diagonal control）
- 正式训练尚未启动；必须先通过单测、真实模型 smoke 与 Codex 代码/科学审查。

## 监控字段

每次完整 eval 记录 epoch、mAP、R1、R5、R10；同时记录：

- 唯一 main PID 与 DataLoader workers；
- GPU 占用；
- `residual_scale`；
- pose coefficient abs mean；
- visibility mean；
- dynamic delta RMS；
- NaN / Inf / Traceback / OOM 扫描。

`< epoch 60` 只记录轨迹，不提前判负。
