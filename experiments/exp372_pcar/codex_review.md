# exp372 PCAR Codex 多路审查

## 审查范围

三路独立只读审查分别覆盖：

1. pose-conditioned attention / pose×CLIP / ReID 直接近邻；
2. 官方 CLIP-ReID 架构、checkpoint 口径与最小插入点；
3. 机制级红队、函数族归约、控制设计与错误门禁风险。

所有审查均未修改 tracked code，未启动训练，未使用 Claude。

## 一致结论

三路均确认：PCAR 在工程上可实现；但两路独立给出新颖性 Gate 直接 NO-GO，另一条代码审计也只允许“进入严格 screen”，明确不能声称新颖性成立。综合 Goal 的预注册规则——“若只剩普通 additive pose bias 或模块拼接，直接 NO-GO”——最终采用更严格裁决：**不进入 screen。**

## 阻断项

### Blocker 1：canonical subtraction 不扩大函数族

`B(P)-B(Pcanonical)` 可直接归约为实例 pose bias 与固定静态 bias 的和。它可作为优化约束或可解释中心，但不是新的 attention 运算。

### Blocker 2：核心操作已有直接近邻

- PeVL：pose mask 调制 CLIP visual attention；
- PAAB：pose-pair mask 进入 ViT attention logits并残差写回；
- MUVA：ReID 中动态 body-part mask逐层进入 CLIP ViT self-attention；
- KPR/PAFormer/ProFD：分别覆盖 pose-conditioned encoder、pose-supervised cross-attention、CLIP part decoder。

因此“pose 改写 CLIP attention”与“CLIP + structural attention”都不能作为 headline。

### Blocker 3：实例姿态燃料证据不足

同一 exp336 checkpoint 的 Gate B：

- global `58.9908`
- correct `59.8357`
- canonical `59.7374`
- shuffled `59.8037`

correct 只比 shuffled/canonical 高 `+0.0320/+0.0984 mAP`。这不否定局部结构分支有用，但不能支撑“instance-relative pose residual 是关键新燃料”。

## 对原六臂门禁的修正

红队指出 frozen `mAP +0.5` 不应作为方法 kill-switch：zero-init frozen adapter严格等于 baseline，手工 alpha 又可能制造分布外 logit 扰动。若未来出现真正不可归约的新机制，正确顺序应是：

1. frozen Gate 只审 parity、attention JS、尺度、等变性、selected/unselected heads 与 NaN/Inf；
2. 性能 Gate 冻结 CLIP，只训练最小 residual projector/head gates；
3. correct-train 与 matched-shuffled-train 做 2×2 train/eval；
4. fixed canonical之外增加 affine-fitted canonical；
5. shuffled 必须是按 pose difficulty/visible-kp-count/body-scale 分箱、无固定点的 cross-image derangement；
6. 不在正式 test 上扫 layer/head/alpha。

这些修正保留为未来方法的实验纪律，不构成继续 PCAR 的理由。

## 最终审查意见

**NO-GO。** 不实现、不训练、不转 layer/head/temperature/query/OT/MoE 小变体。若只想做工程诊断，可另立不承担创新 claim 的实验，但不得混入当前“把 LGPA 改造成我们的创新”目标。
