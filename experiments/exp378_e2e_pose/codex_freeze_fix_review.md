# exp378 hard-freeze 修复审查

## 审查范围

- 失败 execution：commit `5de3b3007b0bd9c5946af47fc79bf85ed10b2e2e`；
- 证据：4090 上 P0/F0 的 e10/e20/e30/e120 checkpoint逐参数比较；
- 代码：`TaskAdaptivePoseField.set_epoch`、生产训练循环、SGD optimizer与 CUDA preflight；
- 约束：不改变 batch、backbone、loss、课程、PSG层数、anchor容量或几何 residual。

## 根因与影响

PyTorch `1.13.1+cu117` 的 `optimizer.zero_grad()`默认 `set_to_none=False`。P0/F0虽然在 e11后
不再给 anchor产生新 objective gradient，但 e10留下的梯度 tensor会被清零而不是置为 `None`；
SGD仍对这些参数应用 momentum与 weight decay。因此“autograd隔离”成立，“参数硬冻结”不成立。

该问题影响 P0/F0 的预注册方法定义，二者旧 execution均不可报告。B0不含 TAPF，D0/J0按设计
持续训练 anchor，不受这项 hard-freeze bug影响。

## 最小修复

`TaskAdaptivePoseField.set_epoch`在且仅在以下条件执行硬冻结：

```text
epoch > E_boot and mode in {F0, P0}
```

动作只有两项：将 anchor参数设为 `requires_grad=False`，并清空残留 `.grad`。optimizer参数组、
scheduler、全局 zero-grad语义均不改变；D0/J0继续保持 anchor可训练。P0 geometry adapter仍由
ReID loss更新，F0 adapter仍不产生更新。

这项修复只冻结 anchor head参数。Stage-2 Swin仍由标准 ReID主路径训练，因此同一冻结 head在变化
的 F2输入上产生的 anchor field仍可能间接漂移；设计文档要求的坐标/方差/置信度漂移审计继续保留。

## 回归审查

新增回归先在 e10用 pose loss建立真实 SGD momentum/weight-decay状态，再调用 e11调度，并故意使用
PyTorch 1.13默认 `optimizer.zero_grad()`。要求 F0与P0各自经过 post-bootstrap optimizer step后，
全部 anchor参数逐位相同；P0的 adapter路径仍可反向传播。

4090冻结运行时已通过以下七项 CPU回归：bootstrap日程、pose-loss梯度隔离、P0 teacher independence、
四臂梯度语义、SGD陈旧 momentum硬冻结、eval teacher independence、四臂初始化一致性。

## CUDA batch64 结果与裁决

exact commit `f1cf1ea70cf39be95e5e8e094430909df61b0739`在4090冻结运行时通过：

- P0 e11：`anchor grad/delta=0/0`，adapter grad=`1.0809e-3`、delta=`3.7460e-6`；
- F0 e11：`anchor grad/delta=0/0`，adapter grad/delta=`0/0`；
- 两臂 pose loss均为 `None`，teacher tensor不进入 post-bootstrap forward，标准 ReID梯度仍到达
  backbone；batch=`64`、AMP init scale=`1024`。

代码差异符合单变量修复范围，审查结论为 `PASS_FOR_FRESH_EXECUTION`。只允许从头启动新的
P0/F0输出目录；旧 checkpoint不得续训，也不得进入正式表格。
