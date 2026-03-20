# 实验 exp119: Common-Support Relational Distillation (CSRD)

## 动机

- `exp047 CSGT` 失败，说明 **只用 overlap 做 pair mining** 不足以把 `cvk_hybrid` 的正信号迁到训练端。
- `exp051 PAML` 中性，说明 **只改 part triplet 的距离形式** 也不足以把 pairwise common-support 几何传给 global embedding。
- `exp109-116` 又说明：把 support 压成 `per-ID EMA prototype` 会损失太多 pair-specific 细节。

因此当前更合理的问题定义不是“如何得到更好的 prototype”，而是：
**如何把 keypoint/common-support 分支已经掌握的 pairwise 比较几何，直接蒸馏给 global embedding。**

## 核心假设

如果用 skeleton branch 的 `CVK-style` pairwise distance 作为 **batch 内 relational teacher**，去约束 global feature 的 pairwise 几何关系，那么：

1. global embedding 会更贴近遮挡场景下真实的 pair comparability
2. 训练后 `global` 模式应优先受益
3. 若机制有效，`equal_concat` 也应随之改善，且对 test-time `cvk_hybrid` 的依赖可能减弱

## 技术方案

相对 `exp030a`，仅新增一个训练项：

- `POSE_CSRD = True`

### CSRD 机制

1. 从 GCN branch 取 `kp_feats` 与 `kp_weights`
2. 在 batch 内构造 **detached teacher distance matrix**
   - teacher 距离使用 `CVK-style` 的 same-keypoint pairwise 距离聚合
   - 权重使用 `sqrt(w_i * w_j)`，与当前强 test-time 距离更一致
3. 用 normalized global feature 构造 **student distance matrix**
4. 对每个 anchor，分别在：
   - 正样本集合
   - 负样本集合
   上做 relational distillation（KL over softmaxed negative distances）
5. `CSRD` 作为额外训练 loss 叠加到原始 `ID + Triplet` 上

### 初始超参

- `POSE_CSRD_WEIGHT = 0.5`
- `POSE_CSRD_WARMUP = 20`
- `POSE_CSRD_TAU = 0.10`

## 对照组

- 主基线：`exp030a`
- 关键失败对照：
  - `exp047 CSGT`：overlap mining 失败
  - `exp051 PAML`：distance alignment 中性
  - `exp110-116`：prototype-bank support-complete 路线天花板已现

## 预期结果

- 理想结果：
  - `global` 模式明显优于 `exp030a-global`
  - `equal_concat` 有稳定正向
  - 后续 `cvk_hybrid` 增益缩小，说明训练端已吸收部分 common-support 几何
- 中性结果：
  - 说明“pairwise teacher 几何”本身不足以重塑 global space
- 负结果：
  - 说明当前 global branch 很难吸收来自 skeleton branch 的 relational teacher，需转向更显式的 pair-conditioned encoder

## 风险与失败解释

1. teacher 来自同一模型分支，可能信息增量不够，最终只形成中性正则。
2. warmup 后 teacher 仍可能偏 noisy，导致 relational distillation 不稳定。
3. 若 `global` 提升而 `equal_concat` 不动，说明 pairwise 几何被学到了，但 fusion 路径未同步受益。
