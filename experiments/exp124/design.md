# 实验 exp124: Stronger Pair-Delta SCRD

## 动机

- `exp121` 最终证明：`stable teacher` 有帮助，但只是 supporting mechanism，不是主突破口
- `exp123` 到 `ep60` 已首次同时超过 `exp119/120` 同阶段，说明 **pair-level teacher-change focusing** 方向本身成立
- 但 `exp123` 也把当前版本的短板暴露得很清楚：
  - `csrd_pd` 长期只有 `0.002~0.003`
  - `csrd_pf` 长期只有 `1.06~1.08`

因此当前最合理的新假设不是再换 teacher，也不是回到 sample-level routing，而是：
**pair-delta focusing 的放大力度太弱，导致正向收益兑现得过慢、过浅。**

## 核心假设

1. `exp123` 的 delayed weak-positive 不是偶然噪声，而是有效信号被弱 focus 稀释后的结果
2. 如果仅提高 `POSE_CSRD_PAIR_WEIGHT_ALPHA`，pair-level focusing 会更明确地放大 teacher-change relations
3. 这种更强的 pair focus 应该比 `exp123` 更早、更清楚地超过 `exp119/120`

## 技术方案

相对 `exp123`，只改一个核心变量：

- `POSE_CSRD_PAIR_WEIGHT_ALPHA: 1.0 -> 4.0`

其余保持完全不变：
- `POSE_CSRD_PAIR_WEIGHT_MODE = 'delta'`
- support-complete teacher 构造不变
- bank 更新规则不变
- `CSRD` 权重 / 温度 / warmup 不变
- backbone、batch size、主损失配比不变

## 对照组

- 直接对照：`exp123 Pair-Delta Focused SCRD`
- 上一层机制对照：`exp120 SCRD`
- supporting 对照：`exp121 SCRD Freeze-30`
- 主基线：`exp030a`

## 预期结果

- 若假设成立：
  1. `exp124` 的 `ep40/50/60` 应优于 `exp123`
  2. 日志中 `csrd_pf` 应明显高于 `1.06~1.08`
  3. pair-level focusing 的收益应更早兑现到 `mAP`

- 若失败：
  1. 说明当前瓶颈不只是 focus 强度不足
  2. 更可能需要改的是 pair 选择方式本身，而不是简单放大 `delta`

## 风险与失败解释

1. 若 `alpha=4.0` 过强，可能把噪声 pair 也一并放大，反而伤 `mAP`
2. 若 `pair_delta` 分布本身就过平，提高 `alpha` 也可能只带来有限变化
3. 若结果变差，说明下一步应考虑 threshold/top-k pair selection，而不是继续线性放大
