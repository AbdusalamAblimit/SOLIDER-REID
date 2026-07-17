# 实验 exp230: BT-PKD on Small (OA-SD baseline)

## 动机
- exp229 在 Tiny 上测试 BT-PKD。本实验在 Small 上测试同一创新。
- Small 有 18 个 Stage 3 blocks (vs Tiny 6)，GSPB 在 Small 上灾难。
- BT-PKD 的平滑 cosine distillation 梯度可能在 Small 上更安全。
- 对照: exp206r (Small OA-SD) = 70.6/82.6

## 核心假设
BT-PKD 的 cosine distillation 梯度比 GSPB 的 CE/triplet 梯度更适合 Small 的 18-block Stage 3。

## 技术方案
与 exp229 相同，仅换 backbone 为 Swin-Small。

## 对照组
- exp206r (Small OA-SD): 70.6/82.6
- exp229 (Tiny BT-PKD): 进行中

## 早停
- ep10 < 30% → 终止
- ep30 < 55% → 终止
