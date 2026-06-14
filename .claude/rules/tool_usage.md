# 新工具使用手册（2026-04 起）

这些工具替代手动 sleep + 轮询模式，用上即可少浪费 context + token。

## Monitor — 流事件监控

**用途**：训练、远程 log、后台任务的关键事件通知。每行 stdout = 一条通知。

**铁则**：
1. `grep --line-buffered` 必须加
2. 过滤器同时覆盖**进度信号**和**崩溃信号**（silence 不等于 success）
3. `tail -F`（大写）比 `-f` 更健壮
4. 长训练用 `persistent: true`，不设 timeout

**训练监控模板**（注意 `Epoch[X]` 无空格）。`<srv>` ∈ {`lab-3090-d`, `lab-4090`, `hyy`}，`<repo>` 为该机仓库路径，见 `remote_server.md`：
```
Monitor(
  persistent=true,
  description="exp{NNN} <srv>",
  command=`ssh <srv> "tail -F /tmp/exp{NNN}.log" | grep -E --line-buffered \
    "Epoch\[[0-9]+\]|mAP:|Rank-1:|Rank-5:|Rank-10:|Traceback|OOM|Killed|Error|RuntimeError|FAILED|NaN|Inf"`
)
```

过滤器保证每 10 epoch 的 `mAP:` eval 会作为独立通知推过来，可以即时更新 monitor.md。

**等条件完成**（直到某文件出现）：
```
Monitor(
  timeout_ms=600000,
  description="wait for <repo>/log/occluded_duke/exp{NNN}/transformer_120.pth",
  command="until ssh <srv> test -f <repo>/log/occluded_duke/exp{NNN}/transformer_120.pth; do sleep 30; done && echo DONE"
)
```

## Bash run_in_background

适合：大 rsync、训练启动、后台 nohup 任务。完成时自动通知。

```
Bash(run_in_background=true, command=`nohup rsync -az ... > /tmp/rsync.log 2>&1`)
```

**不要用**：短命令（< 5s）、需立刻拿结果的命令。

## ScheduleWakeup

`/loop` 动态模式专用。`delaySeconds` 选择：
- 60–270s：cache 保温，适合主动观察训练
- 300s：**避开**（worst of both worlds）
- 1200–1800s：空闲等待的默认

## CronCreate

定时触发 prompt。场景：
- 保持 cache 热（本项目已设 30min 心跳）
- 每 N 小时生成实验进度摘要
- 每日提醒 commit / push

```
CronCreate(cron="*/30 * * * *", durable=true, prompt="heartbeat: check GPUs, nothing required", recurring=true)
```

## TaskCreate + TaskUpdate

每次开工或换大任务时用 TaskCreate 建任务。开始→`in_progress`，完成→`completed`。

**什么时候建任务**：
- 多步并行（≥3 步）
- 跨机器 / 跨实验的 pipeline
- 需要追踪多个后台任务的状态

## 不再 sleep 轮询

旧模式（禁用）：
```
Bash(sleep 120) → Read(log) → Bash(sleep 120) → ...
```

新模式：
```
Monitor(persistent=true, tail -F log | grep --line-buffered ...)
```

Monitor 把 epoch 事件当通知推给我，我只在收到通知时才处理 → 省大量 context 和 token。

## 持续执行的含义

Monitor 运行时 = 已经"在监控"。不需要另外 sleep 等训练结束。**GPU 空闲才是"没活干"**，这时立刻开下一个实验或补文档。

Monitor 跑着不代表"空闲"，但也不代表"不能做别的事"。Monitor 是事件驱动的，两个 epoch 之间完全可以审其他代码、写其他 design.md、做 agent review。
