# 实验 exp130: Residual-KL SCRD

## 动机

- `exp129` 的 Claude 审查指出了一个关键混淆：
  当前 `residual` 写法同时改变了
  1. target 形式
  2. loss family（KL -> Smooth L1）
  3. scale normalization
- 更严重的是，`dist_s - dist_base` 与 `dist_sc - dist_base` 在 Smooth L1 下对 `dist_s` 的梯度方向会把 `dist_base` 抵消掉，因此它不能真正回答 “target dilution” 假设。

因此 `exp129` 不能作为主线证据继续使用。

当前更合理的下一跳是：
**保持 KL-based relational distillation 不变，只把 teacher/student 对齐对象改成 residual relation。**

## 核心假设

1. `exp120/123/125` 的弱点不在于 `CSRD` 这种 KL 形式本身，而在于它一直在蒸馏完整 teacher 几何
2. 若改为对齐
   - `dist_sc - dist_base`
   - `dist_s - dist_base`
   的 soft relation distribution，同时保持 `tau`、KL 和 `delta_top` 不变
3. 那么这才是对 “support-complete 新增 correction 是否被完整 target 稀释” 的有效检验

## 技术方案

相对 `exp125`，仅新增一个真正对应 target 的变量：

- `POSE_CSRD_TARGET_MODE: 'full' -> 'residual_kl'`

保持不变的部分：
- loss family 仍为 KL-div relational distillation
- `tau = 0.10`
- online support-complete teacher 不 freeze
- `pair_weight_mode = delta_top`
- `pair_top_ratio = 0.25`
- backbone / batch size / 主损失全部不变

### Residual-KL CSRD

对每个 anchor 的正/负 pair 子集：

1. `base logits = -(dist_base) / tau`
2. `teacher residual logits = -(dist_sc - dist_base) / tau`
3. `student residual logits = -(dist_s - dist_base) / tau`
4. 对 residual logits 做与 `exp125` 相同的 KL relational distillation

这样改动后：
- 仍保留 soft distribution supervision
- 仍保留 `tau`
- 只把 “要蒸什么” 从 full relation 改成 residual correction

## 对照组

- 直接对照: `exp125 Sparse Pair-Delta SCRD`
- 审查失效对照: `exp129 Residual-Correction SCRD`
- 机制对照: `exp123 Pair-Delta Focused SCRD`
- 主基线: `exp030a`

## 预期结果

如果假设成立：

1. `ep30/40` 起应比 `exp125` 更能兑现 mAP
2. 且结果可以更干净地归因于 “target 从 full 改成 residual”
3. 不再混入 loss family / normalization 改动

如果失败：

1. 说明当前瓶颈不在 full-vs-residual target
2. 更可能需要改 pair signal 本身，或重新定义 student target

## 风险与失败解释

1. residual logits 可能比 full logits 更稀疏、更噪，导致 early-CSRD 段波动
2. 若结果不如 `exp125`，说明完整 teacher distribution 仍提供了必要的整体排序先验
3. 若结果和 `exp125` 近乎等价，则说明 target dilution 不是当前主瓶颈
