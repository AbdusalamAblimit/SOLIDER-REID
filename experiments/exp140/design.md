# 实验 exp140: Confidence-Calibrated LPCS

## 动机

`exp135` 的 full-pair `LPCS` 是有效线，但长期呈现：

1. `mAP` 能涨
2. `R1` 不够稳
3. `exp138` 的 rank-decay 只能小幅缓解，没把主线推强

这说明当前问题可能不是：
- pair 不够多
- rank 强调不够平滑

而是：

**当前 pair correction 虽然会修正，但它不会判断“这次修正到底该不该被信任”。**

## 核心假设

如果让 `LPCS` 不只预测一个残差修正 `delta`，而是同时预测：

1. `raw_delta`: 想修正多少
2. `conf`: 这次修正该被信任多少

并用 support-complete teacher 诱导 `conf` 对齐“teacher 实际改动有多大”的 soft target，那么：

1. 小改动、低必要性的 pairs 会被自动抑制过修正
2. 高必要性的 pairs 仍能保留较强 correction
3. 有机会在不牺牲 `mAP` 的前提下改善 `R1`

## 技术方案

相对 `exp135`，只改一个核心变量：

- `POSE_LPCS_HEAD_MODE: 'residual_conf'`

具体机制：

1. 保留 `pair_mode=all`
2. 保留 `rank_mode=all`
3. 保留 support-complete teacher bank
4. 将原单头 `PairResidualScorer` 改成双输出：
   - `raw_delta`
   - `conf`
5. 最终修正为：
   - `delta = conf * raw_delta`
   - `final_dist = base_dist + delta`
6. 对 `conf` 增加一个 soft calibration 目标：
   - 来自 `pair_change = |teacher_dist - base_dist|`
   - 映射为 `[0, 1]` 的 soft target
7. 训练时额外记录：
   - `lpcs_rdm`: raw delta mean
   - `lpcs_cf`: confidence mean
   - `lpcs_ctm`: confidence target mean
   - `lpcs_cl`: confidence calibration loss

## 对照组

- 直接对照：`exp135 Corrected LPCS`
- 间接参考：`exp138 Rank-Decayed LPCS`
- 并行主候选：`exp139 Query-Context LPCS`

## 预期结果

理想形态：

1. 相对 `exp135`
   - `R1` 更稳或更高
   - `mAP` 不明显掉
2. 如果成立，说明当前瓶颈不是“如何挑 pair”，而是：
   - **pair correction 需要自带 confidence calibration**

## 风险与失败解释

1. 如果和 `exp135` 完全等价：
   - 说明过修正不是当前主瓶颈
2. 如果明显变差：
   - 说明用 teacher change 去监督 confidence 这件事过于粗糙
3. 如果 `lpcs_cf` 长期接近 `0.5` 且 `lpcs_cl` 不降：
   - 说明 confidence head 没学到任何有效校准
