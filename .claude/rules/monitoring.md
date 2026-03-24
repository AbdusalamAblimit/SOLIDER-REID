# 训练监控协议

## 监控节奏

sleep 间隔**严格不超过 5 分钟**：

- Epoch 1-5（前期关键期）：每 **2 分钟**检查
- Epoch 6-30（收敛观察期）：每 **3 分钟**检查
- Epoch 30+（稳定期）：每 **5 分钟**检查

后台训练用 `run_in_background` 或 `nohup` 启动。绝不阻塞等待训练完成。

## 每次检查记录模板

追加写入 `experiments/exp{NNN}/monitor.md`：

```markdown
### [HH:MM:SS] 检查点 #{N}

**状态**: 正常 / 关注 / 异常
**进度**: Epoch {X}/{Total}

| 指标 | 当前值 | 趋势 |
|------|--------|------|
| Total Loss | | |
| ID Loss | | |
| Triplet Loss | | |
| LR | | |
| GPU Mem/Util | | |

**观察**: {一句话}
**决策**: 继续 / 需要干预
```

**每次查看日志必须记录。不允许查看但不记录。**

## DataLoader Worker 识别

PyTorch DataLoader fork `NUM_WORKERS` 个子进程（默认 8），在 `ps aux` 中也显示为 `python train.py`。

- **绝不能 kill worker 进程** — 会导致主进程 crash
- 主进程识别：CPU 占用最高、启动时间最早
- 终止训练：只 kill 主进程 PID，worker 自动退出

## 异常自动干预

| 触发条件 | 操作 |
|----------|------|
| NaN/Inf loss | 立即 kill → 回退 checkpoint → LR 降 0.5 倍重启 |
| loss 突增 5 倍 | 连续观察 3 次（~15 分钟），持续则终止记录 |
| OOM | 精简模块参数或减小输入分辨率。**禁止改 batch size** |
| 进程被 kill/僵死 | 检查 `dmesg | tail`，调整后重启 |
| mAP 停滞 20 epoch | 考虑终止，可先尝试调 LR。终止后先完成文档 |
| mAP 连降 10 epoch | 终止。短期波动正常，连续 10 epoch 才算。先完成文档 |

## 止损规则

- 中期曲线明显持续落后基线 → 记录里程碑对照后及时终止
- 不在同一路线下做多个微调变体
- 止损后流程：记录结论 → 完成文档 → 转文献/代码学习或新机制设计

## 运行环境

- 使用 python 路径：`/root/miniconda3/envs/solider-reid/bin/python`
- 必须设置 `PYTHONUNBUFFERED=1`（`conda run` 会缓冲 stdout）
- 训练 log 要详细输出各 loss 分量（global ID, part ID, global triplet, part triplet 等）
- CHECKPOINT_PERIOD=20，每 20 epoch 保存一次
