# 实验 exp127: SCRC（Support-Conditioned Residual Completion）

## 动机

`exp109` 的 oracle support bank 给出了极强上界，说明 **single-image support incomplete** 不是伪问题。  
但已有两条训练端兑现方式都不够理想：

1. `exp110-115` 的 `SCKD` 只是弱正向，说明“只加蒸馏 loss”太间接
2. `exp116` 的 `SCFR`（直接 hard replace）与 `SCKD` 基本等价，说明“直接替换”又太硬，既切断了原始 low-vis feature 的优化路径，也可能引入 prototype / instance 分布错位

因此下一步不该继续扫 `bank threshold / alpha / ratio`，而应测试：

**能否让 support-complete prior 以“可学习残差”的形式进入 keypoint branch，而不是 hard replace 或 loss-only。**

## 核心假设

对于低可见 keypoint，same-ID support prototype 应该被当作 **residual prior**，而不是绝对真值。

如果模型自己学习一个融合系数：

`kp_completed = kp + gate(kp, proto, score, proto_conf) * (proto - kp)`

那么它有机会同时保留：

1. 当前样本的 instance-specific 线索
2. support-complete prototype 的补全信息
3. 对 low-vis keypoint 的可学习、非硬编码的 completion 强度

这比：

- `SCKD` 的 loss-only 更直接
- `SCFR` 的 hard replace 更柔和

## 技术方案

在 `SkeletonGCNHead` 的 GCN 前新增 `SCRC` 路径：

1. 复用现有 `SupportCompleteBank`
2. 仅对 `score <= low_thr` 且 bank 中已有 support 的 keypoint 生效
3. 从 bank 取出按 visible-norm 缩放后的 prototype
4. 用一个小型 gate MLP 预测每个 keypoint 的融合系数
5. 做 residual completion：
   `kp_feats = kp_feats + gate * (proto - kp_feats)`

gate 输入：

- 当前 keypoint feature
- support prototype feature
- 当前 keypoint score
- prototype confidence

训练时：

- 仍保留 bank 的后台 EMA 更新
- 不计算 `SCKD` loss
- 不做 `SCFR` hard replace

## 对照组

1. 直接对照: `exp116 SCFR`
2. 次对照: `exp110 SCKD`
3. 主基线: `exp030a-eq seed1234`

## 预期结果

如果假设成立：

1. `ep30/40` 起至少不弱于 `exp116`
2. 后期验证应明显优于 `SCFR ≈ SCKD` 的旧天花板
3. 日志中应看到：
   - `scrc_r` 与 `scfr_r` 同量级
   - `scrc_g` 初期较低，后期逐步学习到稳定的非零融合
   - `scrc_dn` 保持稳定而不爆炸

## 风险与失败解释

1. gate 长期接近 0：说明 residual completion 实际未被使用，方法退化成 baseline
2. gate 过大：说明 prototype 过强，可能重新逼近 `SCFR` 的分布错位问题
3. 与 `SCFR` 完全等价：说明瓶颈不在硬替换，而在 prototype 本身的信息上限
4. 比 `SCFR` 更差：说明 low-vis keypoint 的训练更需要稳定 teacher，而不是可学习混合
