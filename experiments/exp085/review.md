# exp085 PAT 审查报告

## 审查范围
- `experiments/exp085/design.md`
- `experiments/exp086/design.md`
- `experiments/exp086/monitor.md`
- `configs/occluded_duke/pose_psg_gcn_paa_parallel.yml`
- `datasets/pose_dataset.py`
- `processor/processor.py`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | HIGH | exp085 目录 | `exp085` 只有设计文档，没有专属 config、monitor、log；真正的并行增强代码与运行记录落在了后续的 `exp086` | 未修复 |
| 2 | HIGH | design.md vs 实现 | 原始 PAT 设计是“原图 + crop 遮挡 + mandatory erasing”，而实际并行增强实现是“full + ROA + heavy random erasing”，没有 crop 分支 | 未修复 |
| 3 | MEDIUM | design.md vs pose_dataset.py | 设计还写了 pose-guided erasing，但并行增强第三路实际调用的是 `_random_erase()`，不是 `_pose_guided_erase()` | 未修复 |

## 审查通过项

- 仓库里后来确实落地了一个 parallel augmentation 训练框架

## 结论

❌ **不通过**

`exp085` 不是一个已完成的独立实验。若后续要重做，建议直接新开编号，把“crop PAT”与实际落地的“parallel ROA + heavy RE”彻底拆开。
