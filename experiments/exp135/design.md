# 实验 exp135: Corrected LPCS Clean Rerun

## 动机

`exp133 LPCS` 原本 intended 要验证：

1. `alpha-fusion` 不够强之后
2. 直接学习 pair-specific residual correction score
3. 是否能把 learned pair correction 真正变成超过固定 `cvk_hybrid` 的方法线

但运行后排查发现，`exp133` 共享接线存在 bug：
- `kp_aux_data` 的构建条件漏掉了 `ltcs_enabled / lpcs_enabled`
- 导致 `LPCS` teacher 特征和 `LPCS` loss 从未真正进入训练

因此 `exp133` 的数值全部不能用于方法判断。`exp135` 的任务不是新想法，而是：

**在修复共享接线 bug 后，第一次真正把 `LPCS` 测起来。**

## 核心假设

如果 `LPCS` 真正激活，那么：

1. 日志中应首次稳定出现 `lpcs_*` 统计
2. `epoch 21+` 后应能看到：
   - `lpcs > 0`
   - `lpcs_dm / lpcs_ds`
   - `lpcs_sm / lpcs_cm / lpcs_wm`
3. 训练曲线和验证形态才有资格被解释为 learned pair correction 的方法证据

## 技术方案

相对 intended 的 `exp133` 方法定义不变，只修共享接线 bug：

1. 保持：
   - `PairResidualScorer`
   - `cvk_residual`
   - support-complete teacher bank
   - full changed-pair weighting
2. 修复 `processor.py` 中 `kp_aux_data` 构建条件
3. 其余配置保持与 `exp133` 原设计一致

## 对照组

1. intended 对照：`exp132 LTCS`
2. clean rerun 对照：后续 `exp136 Sparse LPCS`
3. 历史参考：`exp133` 仅作为失效 run 记录，不作为方法结果对照

## 预期结果

1. 日志中应首次出现完整 `lpcs_*` 统计
2. `ep10/20` 应大体贴近 baseline
3. 真正判别点在 `epoch 21+` 与 `ep30/40`

## 风险与失败解释

1. 如果 `lpcs_*` 仍不出现：
   - 说明共享接线问题还没修干净
   - 这轮实验依然无效
2. 如果 `lpcs_*` 出现但结果显著变差：
   - 说明 `LPCS` 真正激活后确实会扰动主训练
3. 如果 `lpcs_*` 出现且结果转正：
   - 才说明 `LPCS` 这条线第一次被真实验证到
