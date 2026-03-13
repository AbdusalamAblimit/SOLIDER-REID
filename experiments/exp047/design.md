# 实验 exp047: Common-Support-Guided Triplet (CSGT)

## 动机
- `exp039-045` 已经说明：
  - retrieval-time 的 `cvk_hybrid` 能稳定改善 mAP
  - branch 的真正价值不是抬高 global，而是提供 pair-specific common-support
- 但当前训练仍沿用标准 global triplet：
  - 默认 batch 内所有正负 pair 同等可比
  - 没有显式利用共同可见支撑
- 因此下一步不该继续刷 test-time 权重，而应尝试把这条信号迁到训练端。

## 核心假设
- 如果遮挡 ReID 的关键问题真是 **pair 可比性不一致**，那么把 `kp_weights` 构造出的 common-support overlap 写进 triplet mining 后，global 分支应更贴近遮挡场景下的真实可比关系。
- 若该假设成立，训练结果应至少表现为：
  - `exp030a` 同口径评测提升
  - 或后续对 `cvk_hybrid` 的依赖减弱

## 技术方案

### 单变量改动
- 相对 `exp030a`，仅新增一条训练项：
  - `POSE_CSGT = True`
- 训练期默认监控口径仍沿用 `exp030a` 的 `concat_scaled`
- 但最终实验结论必须额外补：
  - `equal_concat`：主汇报口径
  - `global`：机制对照口径

### CSGT 机制
1. 从 skeleton branch 取 `kp_weights`
2. 在 batch 内计算 pairwise common-support overlap
3. 对 global feature 额外计算一条 support-aware triplet：
   - 优先在 overlap 足够高的正负 pair 上做 hard mining
   - 若某 anchor 找不到满足阈值的 pair，则回退到标准 hard mining
4. 原始 global triplet 与 branch loss 保持不变
5. `CSGT` 以独立损失项形式额外叠加，不并入 `wt_g * global_tri + wt_p * part_tri`

### 初始超参
- `POSE_CSGT_WEIGHT = 1.0`
- `POSE_CSGT_MIN_OVERLAP = 0.3`
- `POSE_CSGT_MINE_MODE = both`

## 对照组
- baseline: `exp030a`
- 参考负例：
  - `exp036`：逐关键点独立 triplet 为负，说明“再加局部 triplet”不是答案
- 参考诊断：
  - `exp040/045`：CVK 在测试端稳定转正，说明 common-support 信号真实存在

## 预期结果
- 理想结果：
  - `equal_concat` 或 `global` 指标出现稳定提升
  - 且机制上能解释为更合理的 pair mining
- 中性结果：
  - 训练稳定但最终与 baseline 接近
  - 说明 common-support 更偏向检索时推理，而不是训练期约束
- 负结果：
  - 收敛明显变差或指标下降
  - 说明简单的 support-aware mining 还不足以把 retrieval-time 信号迁进训练

## 最终评测口径
1. 主汇报：`equal_concat`
2. 机制对照：`global`
3. 训练日志中的 `concat_scaled` 只用于中途监控，不用于最终主结论

## 风险与失败解释
1. `kp_weights` 反映的是关键点置信度，不等于真正身份可分性，可能带来错误 pair 选择。
2. 若阈值过高，support-aware mining 会频繁回退，训练信号太弱。
3. 若阈值过低，又会退化成标准 triplet 的近似形式。
