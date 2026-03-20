# 实验 exp122: SGW-SCRD（Support-Gap Weighted SCRD）

## 动机

- `exp109` 已证明：oracle headroom 主要集中在低可见 / support-incomplete 样本
- `exp119` 已证明：pairwise relational teacher 本身成立
- `exp120` 已证明：把 support-complete bank 只用于增强 teacher，机制上确实会让 teacher 更强
- 但 `exp120` 到 `ep90` 仍未优于 `exp119`，说明问题不只是 “teacher 更完整”，而是：
  **support-complete 监督可能被大量本来就不缺 support 的 clean anchor 稀释了**

因此当前最合理的下一步，不是继续增强 teacher，而是：
**只把 relational distillation 聚焦到 teacher 真正发生了 support completion 的 anchor 上。**

## 核心假设

1. `exp109` 的收益并不是均匀分布在所有样本上，而是集中在 support-incomplete 样本
2. `exp120` 之所以没有把更强 teacher 转成更好指标，是因为当前 `CSRD` 对所有 anchor 一视同仁
3. 若把 `CSRD` 的 anchor 权重改为 “该样本被 support-complete teacher 实际补全了多少”，收益应更容易兑现

## 技术方案

### 1. 保持 `exp120` 的 teacher 构造完全不变
- 仍然使用 support-complete teacher bank
- 仍然只替换 low-vis keypoint teacher
- 不改 backbone / batch size / 主 loss / warmup / tau

### 2. 唯一改动：按 sample-level support gap 给 `CSRD` 加权
- 对每个样本计算：
  - `anchor_weight = replace_ratio_i`
  - 即该样本 17 个 keypoint 中，有多少比例真的被 support-complete teacher 替换
- 在 `_compute_csrd_loss` 中：
  - 原来每个 anchor 的 pos/neg KL 等权平均
  - 现在改为用 `anchor_weight` 做加权平均
- 若某个样本没有任何 keypoint 被补全，则该 anchor 不参与 `CSRD`

### 3. 这一步为什么仍是单变量
- 相对 `exp120`：
  - teacher 内容不变
  - bank 更新规则不变
  - loss 形式仍是 `CSRD`
  - 唯一变化只是：**谁该更强地接受这份 support-complete relational supervision**

## 对照组

- 直接对照：`exp120 SCRD`
- 上一层机制对照：`exp119 CSRD`
- 主训练基线：`exp030a`

## 预期结果

- 若假设成立：
  1. `exp122` 应优于 `exp120`
  2. 最先改善的应是 `global` 相关几何，同时 `equal_concat` 也更有机会被拉起
  3. 日志中应看到：
     - `csrd_ar` 小于 1，说明 supervision 不再均匀打给所有 anchor
     - `csrd_aw` 处于合理范围，而不是接近 0

- 若失败：
  1. 说明问题不在于 support-complete supervision 被 clean 样本稀释
  2. 更可能要改的是 distillation target 的结构，而不是 teacher 使用范围

## 风险与失败解释

1. 若 `anchor_weight` 太稀疏，`CSRD` 可能退化成几乎不起作用
2. 若 `exp120` 的问题是 teacher 本身不够可靠，而不是 supervision dilution，那么加权不会带来改善
3. 若 `replace_ratio` 不能准确反映样本级 support gap，后续可能要换成 `low_ratio` 或更强的 pair-level weighting
