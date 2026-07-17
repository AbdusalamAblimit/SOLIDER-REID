# 实验 exp138: Rank-Decayed LPCS

## 动机

`exp137 Hard-Rank LPCS` 已经给出很清楚的负边界：

1. `ranking-aligned` 方向本身没有错
2. 但“只保留 hardest 25% pairs”太激进
3. 它会稳定伤害 `R1`

这说明当前更合理的下一步不是更硬的选择，而是：

**保留 full-pair 上下文，同时对 top-ranked mistakes 做更平滑的强调。**

## 核心假设

如果把 `exp135` 的全 pair `LPCS` 从“均匀聚合”改成“按 rank 位置连续衰减加权”，那么：

1. 可以保留 full-pair 的稳定性
2. 又不会像 `exp137` 那样因为 hard selection 过强而伤害 `R1`
3. 有机会得到比 `exp135` 更好的 `R1`，同时保持 `mAP`

## 技术方案

相对 `exp135`，只改一个核心变量：

- `POSE_LPCS_RANK_MODE: 'rank_decay'`
- `POSE_LPCS_RANK_TAU: 8.0`

具体机制：

1. 保留 `pair_mode=all`
2. 不删除任何 routed pairs
3. 对每个 anchor 内的：
   - positive：按当前 `final_dist` 从大到小排序，越 hard 权重越大
   - negative：按当前 `final_dist` 从小到大排序，越 hard 权重越大
4. 用连续衰减因子 `exp(-rank / tau)` 调整 pair 权重

新增日志：

- `lpcs_rwm`: rank weight mean  
  用于确认当前不是 hard selection，而是真正的连续 rank-decay weighting

## 对照组

- 直接对照：`exp135 Corrected LPCS`
- 负边界对照：`exp137 Hard-Rank LPCS`

## 预期结果

理想形态：

1. 相对 `exp137`
   - `R1` 明显回升
2. 相对 `exp135`
   - `R1` 至少不更差
   - `mAP` 尽量保持

## 风险与失败解释

1. 如果与 `exp135` 完全等价：
   - 说明当前瓶颈不是“hard or soft rank emphasis”
2. 如果仍然伤害 `R1`：
   - 说明单纯基于当前 rank 的加权仍然过于局部
3. 如果 `lpcs_rwm ≈ 1.0`：
   - 说明 rank-decay 退化，必须先排查实现
