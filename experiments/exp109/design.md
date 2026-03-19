# 实验 exp109: Oracle Support Bank 上界诊断

## 动机

- `exp107/108` 已基本否定 retrieval-time `ambiguity/confuser penalty` 主线
- 但 `SGCFR` 明确证明：**跨图 support recovery** 确实能带来大增益
- 训练端 recovery 系列 (`exp091/092/101/105/106`) 之所以没有成功，核心问题很可能不是“recover 这个想法错了”，而是：
  **batch 内没有足够稳定的 same-ID support**
- 因此，在真正实现训练版 support bank distillation 之前，先做一个 oracle 诊断：
  **如果给模型一个 GT same-ID 的 per-keypoint support bank，`cvk_hybrid` 到底还有多少 headroom？**

## 核心假设

1. 若单图表征真的受限于“support 不完整”，那么用同 ID 多图构造 oracle prototype 后，matching 应明显优于原始 `cvk_hybrid`
2. 收益应主要集中在：
   - `multi`
   - `clean multi`
   - 低可见性 query
3. 若 oracle 上界都很小，则说明 training-time support-complete distillation 很难成为主线，应立即止损

## 技术方案

### 1. 特征提取
- 基于 `exp030a` checkpoint
- 提取：
  - `global_feat`
  - `kp_feats`
  - `kp_weights`

### 2. Oracle support bank 构造
- 在 query+gallery 范围内，用 GT `pid` 构造每个 identity、每个 keypoint 的 prototype
- prototype 只由高可见性 keypoint 聚合
- 对当前样本做 leave-one-out，避免把自身直接抄回去

### 3. 恢复方式
- 对低可见 keypoint，用对应 ID 的 prototype 替换其 `kp_feat`
- 比较两种版本：
  - `oracle_feat_only`: 只替换 feature，保留原始 `kp_weight`
  - `oracle_feat_weight`: 替换 feature，并把该 keypoint 的 weight 提升到 prototype confidence

### 4. 评估
- 主基线: `exp030a cvk_hybrid`
- 比较：
  - `base_cvk_hybrid`
  - `oracle_feat_only_cvk`
  - `oracle_feat_weight_cvk`

## 对照组

- `exp030a cvk_hybrid`

## 预期结果

- 如果 oracle 明显为正：
  - 说明 “support-complete teacher” 这条训练主线有真实 headroom
  - 可继续做最小训练版 prototype distillation
- 如果 oracle 只有极小增益或直接负面：
  - 说明 SGCFR 的收益更接近 retrieval-time 邻域特性，而不是可蒸馏到单图编码器的潜在表征
  - support bank 训练线应止损

## 风险与失败解释

1. GT same-ID prototype 也不提升：
   - 说明当前 `kp_feats` 本身不具备可恢复性
2. 只替换 feature 有用，但替换 weight 无用：
   - 说明关键在 latent feature completion，而不是 visibility correction
3. 只在极少数 query 上提升：
   - 说明该方向可能只适合作为 case-specific trick，不足以支撑主线
