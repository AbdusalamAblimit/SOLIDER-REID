# 实验 exp045: 基于重建 `seed42` checkpoint 的 CVK 复核

## 动机
- `exp040` 已在 `exp030a` 原始主 checkpoint 上复核出：
  - `equal_concat` = `61.1% mAP / 73.7% R1`
  - `cvk_hybrid` = `61.9% mAP / 73.2% R1`
- `exp041-043` 又补上了权重敏感性、query 级差分和定性样例。
- 但这些证据仍主要集中在单个 checkpoint 上。
- `exp044` 已成功重建 `exp030a seed42` checkpoint，因此现在应先把第二个 seed 的测试端复核补上。

## 核心假设
- 如果 `cvk_hybrid` 的收益模式具有跨 seed 稳定性，那么在重建出的 `seed42` checkpoint 上，相对同 checkpoint 的 `equal_concat`，它应继续表现出：
  - mAP 持平或更高
  - R1 持平或小幅波动
- 如果在 `seed42` 上完全转成双负，则说明当前 retrieval-time story 的稳定性不足，至少不能再把现有证据写得过强。

## 技术方案

### checkpoint
- 固定使用：`log/occluded_duke/exp044_exp030a_seed42_rebuild/transformer_120.pth`
- 说明：这是按 `exp030a` 配方重建出的 `seed42` 资产，不是历史原始文件直读；因此结论应写成“重建 seed42 复核”。

### 子实验
- `045a`: `equal_concat`
  - 目的：给当前重建 `seed42` checkpoint 生成直接对照口径
- `045b`: `cvk_hybrid`
  - 目的：验证 `cvk_hybrid` 是否能在第二个 seed 上继续保留正向 mAP 信号

### 评测设定
- Backbone 固定为 `Swin-Tiny`
- batch size 不变
- 不改训练参数
- 唯一变量为测试特征模式
- 每个子实验使用独立 `OUTPUT_DIR`

## 对照组
- 直接对照：`045a equal_concat`
- 待验证项：`045b cvk_hybrid`
- 参考背景：
  - `040a/040b`：原始主 checkpoint 上的 `+0.8% mAP / -0.5% R1`
  - `exp044`：重建 `seed42` checkpoint 训练完成，epoch120 默认监控口径为 `concat_scaled = 59.8% / 72.9%`

## 预期结果
- 理想结果：`045b` 相对 `045a` 继续给出正 mAP
- 中性结果：`045b` 与 `045a` 基本持平
- 负结果：`045b` 明显低于 `045a`

## 风险与失败解释
1. `exp044` 使用的是重建 checkpoint，因此和历史多 seed 表中的 seed42 数值不必逐点完全相同。
2. 若 `045a` 与历史 seed42 记录存在偏差，应优先以当前重建资产的测试日志为准，不凭旧表回填。
3. 即使 `045b` 为正，它仍然是 retrieval-time 证据，不能被写成训练端创新。
