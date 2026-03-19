# 实验 exp113: SCKD 统计诊断（UPDATE_THR=0.7）

## 动机

用户明确指出一个关键风险：`SCKD` loss 在 `exp110/111/112` 中并没有明显下降。这件事不能轻描淡写，因为它关系到当前主线到底是在学习有意义的 support-complete 对齐，还是仅仅在施加一个几乎恒定的正则项。

当前日志只打印一个裸 `sckd` 标量，信息远远不够。`sckd` 不下降可能有两种完全不同的解释：

1. **坏解释**：蒸馏本身没学到，teacher-student cosine 长期不改善；
2. **正常解释**：随着 bank 覆盖率扩大、更多困难 low-vis keypoint 被纳入蒸馏，raw loss 保持平或略升，但 teacher 对齐实际上在改善。

因此必须补机制统计，而不是继续盲猜。

## 核心假设

如果 `UPDATE_THR=0.7` 的当前正信号是可信的，那么在更细的统计上应看到：

1. `sckd_cos` 随训练逐步改善；
2. `sckd_eligr / sckd_actr` 随 bank 成熟而变化；
3. `sckd_conf / sckd_count` 能解释为何 raw `sckd` 不明显下降。

反之，如果 `sckd` 长期不降，且 `cosine` 也不改善、覆盖率也不变，那就说明当前蒸馏机制本身存在问题。

## 技术方案

1. 在 `SupportCompleteBank.compute_loss` 中新增统计：
   - `sckd_pairs`
   - `sckd_lowr`
   - `sckd_actr`
   - `sckd_eligr`
   - `sckd_conf`
   - `sckd_count`
   - `sckd_cos`
2. 在训练日志中一起打印这些统计
3. 复跑当前更有希望的配置：
   - `POSE_SCKD_UPDATE_THR = 0.7`
   - 其余保持与 `exp112` 一致

## 对照组

- 机制对照：`exp112`（同配置但无统计）
- 方法对照：`exp110`

## 预期结果

1. 若主线有效，则 `sckd_cos` 应比 raw `sckd` 更能反映训练进展；
2. 若 `raw sckd` 不降但 `sckd_cos` 改善，同时 `elig_ratio` 上升，则“loss 不降”并非致命问题；
3. 若所有统计都停滞，则需要重新审视蒸馏目标。

## 风险与失败解释

1. 统计本身不改变方法，只改变解释力；它不能直接带来涨点
2. 若日志显示机制确实没学到，那当前主线必须马上转向更强的 teacher/student 定义，而不是继续扫阈值
