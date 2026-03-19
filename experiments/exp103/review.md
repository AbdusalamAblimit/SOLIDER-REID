# exp103 审查报告

## 审查范围
- `experiments/exp103/design.md`
- `configs/occluded_duke/pose_psg_gcn_paa_roa_sgmt.yml`
- `log/occluded_duke/exp103_roa_sgmt/train_log.txt`
- `datasets/`
- `processor/`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | HIGH | design.md vs config/log | `exp103` 设计写的是 PGCE（batch 内跨图人体遮挡），但当前真正存在的 config 和日志是 `ROA + SGMT` 组合实验，实验编号与内容完全错位 | 未修复 |
| 2 | HIGH | repo-wide | 仓库里没有任何实现会“从 batch 内另一张图裁身体区域再粘贴回来”，PGCE 从未真正落地 | 未修复 |
| 3 | MEDIUM | exp103 目录 | 没有 `monitor.md` 等文档来解释为什么 `exp103` 最终跑成了 `roa_sgmt` | 未修复 |

## 审查通过项

- 当前 `pose_psg_gcn_paa_roa_sgmt.yml` 作为一个普通组合 config 是可运行的

## 结论

❌ **不通过**

`exp103` 不能作为 PGCE 的任何证据。若后续要继续这个方向，应重新开干净编号或先把编号/文档纠正。
