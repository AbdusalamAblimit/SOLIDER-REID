# 实验 exp128: Exact Top-K Pair SCRD + Freeze30

## 动机

当前证据已经形成了两条相互独立的正信号：

1. `exp121` 证明 `stable teacher (freeze30)` 对 `SCRD` 是持续弱正向
2. `exp125` 证明结构化 pair focus 是当前最有生命力的主线
3. `exp126` 则在机制上首次真正实现了 exact top-k 的“真稀疏 routing”

与此同时，`exp127 SCRC` 到 `ep100` 已经暴露出：

- gate 几乎塌成 `1.0`
- 指标未超过 `SCFR/SCKD`

说明当前本地不该继续投入到 feature-level bank completion，而应回到更有希望的 relational 主线。

## 核心假设

如果 `exp126` 的 exact top-k 稀疏路由方向本身成立，那么把 `exp121` 已验证有帮助的 `freeze30 stable teacher` 接到 exact-topk 版本上，应该比 online teacher 更稳。

## 技术方案

基于 `exp126`，只改一个变量：

- `POSE_CSRD_ST_UPDATE_STOP_EPOCH: -1 -> 30`

其余全部保持与 `exp126` 相同：

- `delta_top_exact`
- `pair_top_ratio = 0.25`
- `pair_weight_alpha = 1.0`
- support-complete teacher 其余配置不变

## 对照组

1. 直接对照: `exp126`
2. 次对照: `exp125`
3. supporting 对照: `exp121`

## 预期结果

如果假设成立：

1. `ep20` 前后与 `exp126` 基本重合
2. `epoch 30+` 后在 late-stage 验证上比 online teacher 更稳
3. `pair_select_ratio` 仍应保持 exact-topk 的真稀疏水平，而不是回到 `0.90+`

## 风险与失败解释

1. 若比 `exp126` 更差，说明 exact sparse routing 对 teacher 的“新鲜度”依赖更高，不适合 freeze30
2. 若仅完全等价，说明 stable teacher 在 true sparse regime 中只是弱辅助，不值得进入 full model
3. 若明显更强，则说明当前最有希望的 full-model 方向是：
   **exact sparse routing + stable support-complete teacher**
