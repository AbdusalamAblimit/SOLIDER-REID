# 实验 exp120: SCRD（Support-Complete Relational Distillation）

## 动机

- `exp109` 证明：只要 teacher 真正 support-complete，headroom 很大
- `exp110-116` 证明：把 support 压成 `per-ID prototype` 后，直接做 pointwise 蒸馏不够强
- `exp119` 证明：**relational distillation 这件事本身是对的**
  - `global = 60.4 / 70.3`，相对 `exp030a-g seed1234 = 59.8 / 69.9` 为 `+0.6 / +0.4`
  - 但 `equal_concat` 仍只到 `61.1 / 73.2`，说明 teacher 本身还不够强

因此当前最合理的下一步，不是扫 `CSRD` 权重/温度，也不是回到 prototype-pointwise loss，而是：
**把 support-complete bank 只用于增强 `CSRD teacher`，而不再直接约束 student。**

## 核心假设

1. `exp119` 的瓶颈不是 relational distillation 无效，而是 teacher 仍来自单图 `kp_feats`，本身不完整
2. 如果用 same-ID support bank 补全 low-vis keypoint teacher，再做 relational distillation，收益应强于 `exp119`
3. 相对 `exp119`，最先被拉起的仍应是 `global`；若方法真的更贴近 `exp109`，则 `equal_concat / cvk_hybrid` 也应开始同步受益

## 技术方案

### 1. 保留 `exp119` 的 student 与 loss 形式
- 仍然使用 `CSRD`
- student 仍是 `global embedding`
- 主 loss、backbone、batch size 全不改

### 2. 新增 support-complete teacher enhancement
- 维护一个 `SupportCompleteBank`
- bank 仅由高可见 keypoint 更新：
  - `kp_weight >= 0.7`
- 在 `epoch > 20` 且 bank 可用时：
  - 对当前 batch 的 low-vis keypoint teacher 做替换：
    - `kp_weight <= 0.3`
    - 有 same-ID prototype 支持时，用 prototype 补全 teacher keypoint feature

### 3. 仅增强 teacher，不直接蒸馏点特征
- 不再像 `SCKD` 一样把 student keypoint 直接拉向 prototype
- 只把补全后的 teacher keypoint feature 用于构造 `CSRD` 的 relational teacher distance
- teacher weight 仍保持原始 `kp_weights`
  - 对应 `exp109` 中 `oracle_feat_only` 的更干净版本

## 对照组

- 直接对照：`exp119 CSRD`
  - `equal_concat = 61.1 / 73.2`
  - `global = 60.4 / 70.3`
  - `cvk_hybrid = 62.0 / 73.2`
- 主基线锚点仍为 `exp030a`

## 预期结果

- 若假设成立：
  1. `global` 应继续高于 `exp119`
  2. `equal_concat` 不应再只是近乎持平，而应开始出现更清楚的正向
  3. 日志中 `csrd_sr / csrd_sn` 应显示 support-complete teacher 确实在工作

- 若失败：
  1. 说明“teacher 不够 support-complete”不是当前主瓶颈
  2. 更可能的问题是 relational objective 仍然过弱，或 student 目标空间选错

## 风险与失败解释

1. bank 仍可能引入 prototype drift，只是这次 drift 体现在 teacher 质量，而不是 pointwise loss
2. 若 `csrd_sr` 很低，说明 teacher enhancement 实际覆盖太少
3. 若 `global` 继续涨但 `equal_concat` 仍不动，说明最终问题可能转移到 fusion / student space，而不只是 teacher 质量
