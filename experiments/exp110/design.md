# 实验 exp110: SCKD（Support-Complete Keypoint Distillation）

## 动机

- `exp109` 的 oracle support bank 诊断给出极强 headroom：
  - `base_cvk_hybrid = 61.88 / 73.26`
  - `oracle_feat_only_cvk = 66.15 / 77.87`
- 这说明当前缺口里确实存在一块：
  **单图关键点表征缺少 support-complete identity 信息**
- 过去的 recovery 训练为什么失败：
  - `CIPGFR / LSRM / TTSFR` 依赖 batch 内 same-ID support，太弱
  - `SGMT / PISD / PACD` 监督落点更偏 global 或 masking，一直不够准

因此 `exp110` 的目标不是再做 retrieval-time trick，而是做一个**最小训练版**：
用 identity-level prototype bank 把 low-visibility keypoints 蒸馏向更完整的 support teacher。

## 核心假设

1. 若 batch-local recovery 失败的原因真是 support 来源太弱，那么换成持久的 `per-ID / per-keypoint prototype bank` 后，训练端应该更稳定。
2. 若 `exp109` 的 headroom 有一部分可蒸馏到单图编码器，那么在**不改测试流程**的前提下，`equal_concat` 应出现正信号。
3. 第一版只做 low-visibility keypoint distillation，就足以验证这条主线是否成立；不需要一上来叠加 decoder 或 re-rank。

## 技术方案

### 1. Support-Complete Bank
- 新增 `SupportCompleteBank`
- 为每个训练 ID、每个 keypoint 维护一个 prototype
- prototype 只由高可见 keypoint 更新：
  - `kp_weight >= 0.5`
- 更新方式：
  - EMA, `momentum = 0.9`

### 2. Distillation
- 对当前 batch 中的 low-visibility keypoint：
  - `kp_weight <= 0.3`
- 若该 ID 的对应 prototype 已存在：
  - 用 cosine distillation 将当前 keypoint feature 拉向 prototype
- 只对 keypoint feature 做蒸馏，不改 global branch，不改测试时距离公式

### 3. 训练约束
- 基线固定为 `exp030a`
- `POSE_TEST_FEAT = equal_concat`
- 不加 decoder
- 不加 test-time rerank
- 不和其它 recovery/uncertainty 模块叠加

## 对照组

- `exp030a-eq = 60.73% / 72.57%`（3-seed mean）
- 当前首轮只跑单 seed，作为方向判定

## 预期结果

- 若成立：
  - `equal_concat` 至少出现 `+0.5%` 级别正信号
  - 后期曲线不应像 `CIPGFR / LSRM` 那样持续落后
- 若失败：
  - 说明 oracle headroom 很难直接蒸馏到单图编码器
  - support-complete 方向需重写为更强 teacher / recovered pooling，而不是继续在当前最小版小修小补

## 风险与失败解释

1. prototype drift：bank 被早期噪声污染
2. train/test ID gap：训练 ID prototype 无法迁移到测试身份
3. 仅 keypoint 级 cosine distillation 不够，需要 recovered pooled representation 才能把增益传到最终检索特征
