# 实验 exp123: Pair-Delta Focused SCRD

## 动机

- `exp119` 已证明：`pairwise relational teacher` 本身是有效的
- `exp120` 已证明：support-complete teacher enhancement 确实会让 teacher 几何更强
- `exp122` 则进一步证明：问题不是 “哪些样本更该被蒸馏”，因为 sample-level `replace_ratio` selective weighting 已经工作但没有转成收益

因此当前最合理的新假设不再是 sample-level selective supervision，而是：
**support-complete teacher 真正改变的是一部分 pairwise relation，distillation 应聚焦这些 teacher-change pairs。**

## 核心假设

1. `exp120` 的监督之所以没有兑现，是因为 `CSRD` 仍对所有 pair 等权对齐
2. `exp122` 失败说明 sample-level `replace_ratio` 太粗，不能精确定位哪些 relation 真被 support completion 改变
3. 如果改为按 **teacher 几何变化量** 聚焦 pair-level distillation，那么：
   - `global` 空间会更准确吸收 support-complete teacher 的有效部分
   - 中期验证应优于 `exp120`
   - 日志里应出现稳定的 `csrd_pd > 0`、`csrd_pf > 1`

## 技术方案

相对 `exp120`，仅新增一个训练机制：

- `POSE_CSRD_PAIR_WEIGHT_MODE = 'delta'`

### Pair-Delta Focused CSRD

1. 保持 `exp120` 的 support-complete teacher 完全不变
   - bank 更新规则不变
   - low-vis teacher replacement 不变
   - `CSRD` 的 student / tau / loss weight 全不变

2. 在 `CSRD` 内同时构造两份 teacher 距离
   - `dist_base`: 原始单图 keypoint teacher
   - `dist_sc`: support-complete teacher

3. 对每个 anchor 的正/负样本集合，计算：
   - `pair_delta = |dist_sc - dist_base|`

4. 用 `pair_delta` 构造 pair focus：
   - `focus = 1 + alpha * delta / max(delta)`
   - 用同一组 `focus` 同时重加权 teacher/student softmax
   - 本质上是把 distillation 聚焦到 **被 support-complete teacher 真正改变过的 relations**

### 初始超参

- `POSE_CSRD_PAIR_WEIGHT_MODE = 'delta'`
- `POSE_CSRD_PAIR_WEIGHT_ALPHA = 1.0`

## 对照组

- 直接对照：`exp120 SCRD`
- 失败对照：`exp122 SGW-SCRD`
- 上一层机制对照：`exp119 CSRD`
- 主基线：`exp030a`

## 预期结果

- 若假设成立：
  1. `exp123` 应优于 `exp120` 的中期 checkpoint
  2. 相对 `exp122`，应体现出更清楚的正向
  3. 日志中：
     - `csrd_pd` 持续大于 0
     - `csrd_pf` 稳定大于 1

- 若失败：
  1. 说明问题不只是 selective supervision 粒度不够
  2. 更可能需要改的是 teacher 稳定性与 pair focus 的结合方式，例如 `freeze30 + pair-delta`

## 风险与失败解释

1. 若 `pair_delta` 太 noisy，可能把 distillation 过度聚焦到不稳定 pair 上
2. 若 `pair_delta` 分布过平，focus 可能退化成近乎等权
3. 若 `exp123` 仍不优于 `exp120`，说明当前 `SCRD` 的瓶颈更可能在 teacher source / student target，而不只是 supervision routing
