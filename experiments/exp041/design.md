# 实验 exp041: CVK hybrid 权重敏感性

## 动机
- `exp040` 已在 `exp030a` 原始 checkpoint 上复核出：
  - `equal_concat` = `61.1% mAP / 73.7% R1`
  - `cvk_hybrid (1:1)` = `61.9% mAP / 73.2% R1`
- 当前还不能判断 `1:1` 是稳健工作点，还是偶然点位。
- 在进入更多 checkpoint / seed 之前，先做最小成本的权重敏感性验证，可以更快判断这条 retrieval-time reasoning 线是否稳定。

## 核心假设
- 如果 `cvk_hybrid` 的收益来自真实的 common-support 补充，而不是偶然平均，那么在 `global:cvk` 比例略微偏向任一侧时，结果应保持接近，且趋势可解释。
- 若只要偏离 `1:1` 就明显失效，则说明这条线当前仍然很脆弱。

## 技术方案

### 固定条件
- checkpoint: `log/occluded_duke/exp030a_psg_gcn/transformer_120.pth`
- 测试模式固定为 `cvk_hybrid`
- 只改：
  - `TEST.CVK_GLOBAL_WEIGHT`
  - `TEST.CVK_KP_WEIGHT`

### 子实验
- `041a`: `global:cvk = 2:1`
- `041b`: `global:cvk = 1:2`

## 对照组
- 直接参考 `exp040b`:
  - `global:cvk = 1:1`
  - `61.9% mAP / 73.2% R1`

## 预期结果
- 若 `2:1` 更好：
  - 说明 CVK 更适合作为轻量修正项，global 仍应占主导
- 若 `1:2` 更好：
  - 说明共同可见关键点距离的贡献被 `1:1` 低估
- 若两边都接近 `1:1`：
  - 说明该机制存在一定稳健区间，适合继续往多 checkpoint / seed 推进

## 风险与失败解释
1. 当前只测试两个方向，不能直接推出全局最优权重。
2. 若 `2:1` 与 `1:2` 都明显变差，也不等于整条路线失败，只能说明 `1:1` 附近更稳。
3. 这是 test-time 比例敏感性，不应被包装为训练端贡献。
