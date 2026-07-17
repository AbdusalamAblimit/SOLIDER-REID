# exp080 ATF 审查报告

## 审查范围
- `experiments/exp080/design.md`
- `scripts/`
- `model/`
- `processor/`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | HIGH | exp080 全体 | 只有 `design.md`，没有独立评估脚本、没有 config、没有 monitor、没有日志，`ATF` 没有真正落地 | 未修复 |
| 2 | MEDIUM | design.md vs repo | 设计里提到的“单人 query 用 global、多人 query 用 equal_concat、必要时 zero-pad”在仓库里没有任何实现入口 | 未修复 |

## 审查通过项

- 设计目标清楚，且属于纯 test-time 实验，不要求训练侧改动

## 结论

❌ **不通过**

`exp080` 目前只是想法稿，不是一个可审查的已实现实验。若后续要重做，需要先补独立评估脚本和可复现的运行记录。
