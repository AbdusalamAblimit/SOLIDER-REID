# exp099 POT-Match 审查报告

## 审查范围
- `experiments/exp099/design.md`
- `scripts/eval_pot.py`
- `model/modules/sinkhorn_distance.py`
- `processor/processor.py`
- `config/`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | HIGH | eval_pot.py / sinkhorn_distance.py | 当前实现把 query/gallery 权重都归一化到和为 1，做的是标准 balanced Sinkhorn OT，不是设计标题里声称的 Partial OT | 未修复 |
| 2 | HIGH | design.md vs repo | 设计写了“新增 OT triplet loss 进入训练”，但 `OTTripletLoss` 在仓库里没有被 config、processor 或 model 引用，训练侧完全没接上 | 未修复 |
| 3 | MEDIUM | exp099 全体 | 没有 exp099 专属 config、monitor、运行日志；现阶段只有脚本级原型 | 未修复 |

## 审查通过项

- test-time OT 脚本可以实际提取 keypoint features 并计算一张 OT 距离矩阵
- `SinkhornDistance` 的 log-domain 迭代本身是数值上可运行的

## 结论

❌ **不通过**

`exp099` 现在不是“Partial Optimal Transport Matching”这个实验，只是一个 balanced OT 的评估原型。训练版设计也没有落地。
