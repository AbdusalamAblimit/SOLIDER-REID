# 实验 exp234: Tiny OA-SD 240 epochs (延长训练)

## 动机
- 所有创新在 120 epoch 训练下未能超过 baseline
- exp233 per-part 的 id_part 在 ep70 才降到 3.0 (baseline ep30 就<1.0)
- 问题可能是 120 epoch 不够，而不是方法不行
- 延长到 240 epoch 看 baseline 本身是否还在涨

## 核心假设
如果 baseline 在 240 epoch 显著优于 120 epoch，说明当前 120ep 是次优的。

## 技术方案
与 exp191 完全相同配置，仅 MAX_EPOCHS=240。

## 对照组
- exp191 (120ep): 63.2/75.4

## 早停
- ep120 < exp191 (63.2%) → 有问题
