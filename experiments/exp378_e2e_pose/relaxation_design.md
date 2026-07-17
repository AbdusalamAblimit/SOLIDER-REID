# exp378-R：显式零目标梯度 SGD 姿态锚点松弛

## 动机

旧 commit `5de3b30` 的 P0 相对同配方 B0 为 `+1.1 mAP / +1.1 R1`，但 checkpoint
审计证明其 anchor 在 e10 后仍被 PyTorch 1.13 的 SGD momentum 与 weight decay 改写，因而
不能作为 hard-freeze TAPF 的正式结果。修复后严格冻结 P0 仍有 `+0.5 mAP`，但没有 R1 增益，
且 R5/R10 下降。两者的差异说明：bootstrap 后有限延续的优化器状态可能比瞬时硬冻结更适合
完成 teacher 到 ReID-only 的交接。

旧行为不能直接包装成方法，也不能称为单纯 momentum。它同时包含 momentum buffer、weight
decay、当前学习率日程与 AMP overflow 时的整步跳过。因此本实验把它重新实现为显式、无新目标
梯度、可逐步复现的 **zero-objective-gradient SGD relaxation**，并用 matched residual
OFF/ON 对照拆解其贡献。

## 核心假设

e10 时 anchor 已获得有意义的姿态 bootstrap 状态，但直接 hard-freeze 可能在 handoff 边界过早
截断已经建立的低速优化轨迹。若 e11 后：

1. anchor 不再读取 teacher；
2. anchor 不接收 pose 或 ReID objective 的 autograd；
3. 只让 e10 已存在的 SGD state 在零 objective gradient 下自然衰减；

则锚点可发生有限、有方向的平滑松弛，而不会变成自由学习的身份 attention。若该松弛有效，
matched MR-P0/MR-F0 应能区分 anchor transition 本身与 `17×4` geometry residual 的作用。

## 技术方案

### 1. 显式 transition policy

新增默认关闭的配置：

```yaml
MODEL:
  POSE_TAPF_ANCHOR_TRANSITION: 'hard'  # hard | sgd_relax
```

- `hard`：保持 commit `f1cf1ea` 的现有行为；e11 起 anchor `requires_grad=False`、
  `.grad=None`，optimizer 不更新 anchor；
- `sgd_relax`：e11 起同样令 anchor `requires_grad=False` 并断开所有 objective autograd；
  但在每次 `GradScaler.step(optimizer)` 前，显式为 anchor 参数写入零梯度 tensor，使标准 SGD
  按原参数组继续执行：

\[
v_t=\mu v_{t-1}+\lambda\theta_t,\qquad
\theta_{t+1}=\theta_t-\eta_t v_t.
\]

这里 `mu`、`lambda`、`eta_t` 分别沿用原 momentum、参数组 weight decay 与当前 scheduler
学习率。禁止私写另一套参数更新器，以保证 AMP overflow 时 anchor 与主 optimizer 同步跳步。
每次 step 后立即把显式零梯度恢复为 `None`，避免下一轮把其误读为 objective gradient。

该 transition 只允许用于 `F0/P0`。`D0/J0` 持续接收 pose objective，不属于零目标梯度松弛。

### 2. 最小 2×2 对照

与有效 B0 一起形成：

| anchor transition | residual OFF | residual ON |
|---|---|---|
| hard | corrected hard F0 | 已完成 corrected hard P0 |
| explicit SGD relaxation | MR-F0 | MR-P0 |

执行顺序：

1. 当前 fresh corrected hard F0；
2. 同一新 exact commit 的 fresh MR-F0；
3. 同一新 exact commit 的 fresh MR-P0。

必须报告：

- hard 条件下 `P0-F0`：geometry residual 主效应；
- relaxation 条件下 `MR-P0-MR-F0`：相同松弛轨迹上的 residual 主效应；
- `MR-F0-hard F0`：无 residual 时 transition 主效应；
- 差分中的差分：residual 与 relaxation 是否存在协同。

旧 P0/F0 只标记为 `INVALID_AS_HARD_FREEZE / VALID_RELAXATION_PILOT`，不得续训，也不得
进入论文正式性能表。正式 relaxation 结论只来自 fresh matched MR-F0/MR-P0。

## 实现与回归门禁

正式训练前必须同时通过：

1. **逐步 legacy parity**：真实建立 e10 momentum/weight-decay state 后，比较旧 PyTorch 1.13
   零梯度隐式路径与新显式路径，参数及 momentum buffer 单步、多步均逐位一致；
2. **无 objective gradient**：e11 后 teacher tensor 为 `None`、pose loss关闭、anchor
   `requires_grad=False`；显式写零前 anchor `.grad is None`；
3. **AMP 同步**：正常 step 时 anchor有限更新；人为 overflow/skip 时 anchor与主参数均不更新；
4. **matched trajectory**：相同初始化与数据顺序下，MR-F0/MR-P0 的 anchor参数轨迹逐位一致；
5. **residual isolation**：MR-F0 adapter `0/6 changed`，MR-P0 adapter `6/6`有限更新；
6. **部署一致性**：e11 后及 eval 的 `pose_dict=correct/shuffle/None` descriptor exact parity；
7. **数值审计**：每个 checkpoint 记录 anchor parameter max/L2 drift、momentum-buffer norm、
   关节坐标/尺度/置信度漂移、adapter delta 与所有完整 eval；
8. 默认 `POSE_TAPF_ANCHOR_TRANSITION=hard`，非 TAPF 与现有 hard TAPF forward逐元素不变。

## 解释边界与后续

- 成功只能说明“零新目标梯度的优化器状态松弛”改善当前 TAPF handoff；不能声称普通 pose
  momentum、EMA teacher 或自由 pose fine-tuning 的贡献；
- 若 MR-F0 与 MR-P0 都提高，主要收益属于 transition；若只有 MR-P0提高，才支持它与
  geometry residual 的协同；
- 若显式复现失败，旧 `+1.1`保留为执行诊断，不追逐未受控 bug；
- hard/relax `2×2` 完成后再决定 D0/J0/R0与 Gate B的最小集合；
- Hierarchical TAPF 使用本实验选出的 transition policy 作为共同条件，另立独立 design，
  不与本轮单锚点归因混成一个变量。
