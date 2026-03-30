# 实验 exp201: Global SupCon — 在 global feature 上也做 SupCon

## 动机
- 当前 SupCon 只在 per-token (feat[1:]) 上做，global (feat[0]) 只有 CE + triplet
- exp187 最佳 64.9/76.6，瓶颈在 mAP
- Performance ceiling analysis 指出：mAP 要求在 hard negatives 上精确排序
- **假设**: global feature 上也加 SupCon 可以提升 global 的判别力，尤其在 hard negatives 上

## 核心假设
在 global feature (pooled, feat[0]) 上增加 SupCon loss，可以增强 global 特征的 intra-class 紧凑度和 inter-class 可分性，提升 mAP。

## 技术方案
- 当前: SupCon 只在 `feat[1:]` (per-token) 上计算
- 修改: 也在 `feat[0]` (global) 上计算 SupCon
- 实现: 在 `loss/make_loss.py` 中，当 SupCon 启用时，额外对 global feature 计算 SupCon loss
- 新 config: `POSE_STR_SUPCON_GLOBAL = True` (是否也在 global 上做)

### 梯度流
- feat[0] (global): CE + triplet + SupCon_global → 三个梯度合力
- feat[1:] (tokens): CE + triplet + SupCon_token → 三个梯度合力（不变）

## 预期结果
- 假设成立: mAP +0.3-0.5% (global 判别力增强)
- 如果中性: global 上的 SupCon 信号冗余（CE+triplet 已够）
- 如果负向: global SupCon 与 0.5x global loss scale 冲突

## 对照组
- exp187 (SupCon per-token only): 64.9/76.6

## 创新门槛评估
这是一个消融实验（测试 SupCon 的作用范围），不算独立创新。
但它可能提供 mAP 改善，且实验成本低（只改 loss 计算）。
如果有效，说明"multi-level SupCon"是有价值的方向。
