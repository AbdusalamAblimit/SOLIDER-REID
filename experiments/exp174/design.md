# 实验 exp174: Triple Pose Injection + Per-Token SupCon

## 动机
- exp173 triple injection = R1 74.7% (新高)，但 mAP 63.0% (略低于 exp171 的 63.2%)
- 所有实验都用 CE + Triplet 训练目标，从未改变过训练目标本身
- SupCon 直接优化特征相似度（metric space），比 CE (classification boundary) 更适合检索
- 在 triple injection 基础上用 SupCon 替代 per-token CE，可能同时提升 mAP 和 R1

## 核心假设
用 SupCon 替代 per-token CE，直接在特征空间优化 same-ID 聚拢 / different-ID 推开。

## 技术方案
- 保留 global CE（保持分类能力）
- 替代 per-token CE → per-token SupCon（优化特征 metric）
- 保留 per-token triplet（互补：SupCon 全对 vs triplet 最难对）
- temperature = 0.07（SupCon 标准值）
- 基于 exp173 最佳配置（triple injection + PLBOA）

### 与 CE 的关键区别
| | CE | SupCon |
|---|---|---|
| 优化目标 | 分类正确率 | 特征相似度 |
| 利用的信息 | 样本-标签对 | 所有 same-ID 对 |
| 梯度信号 | 与原型距离 | 与所有正负样本距离 |
| 适合 | 分类 | 检索 |

## 对照组
- exp173 (triple + CE): 63.0/74.7
- 消融变量：per-token CE → per-token SupCon
