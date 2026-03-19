# exp090 SGCFR 审查报告

## 审查范围
- `experiments/exp090/design.md`
- `scripts/eval_sgcfr.py`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | MEDIUM | exp090 目录 | 没有独立 monitor / 结果记录，当前只看到脚本实现，没有封装成标准实验工件 | 未修复 |
| 2 | LOW | eval_sgcfr.py | `compute_kp_distance()` 用的是二值 common visibility，而不是更细粒度的置信度加权，和 design 里“weighted matching”的表述略有简化 | 接受 |

## 审查通过项

- 脚本会提取 query/gallery 的 global 与 per-keypoint 特征
- 初始粗排、top-K 候选恢复、重算距离、alpha sweep 的主流程是通的
- 当基础 config 使用 `equal_concat` 时，粗排会直接用该特征，不会错误退化成 global-only
- 整个方法是 test-time only，不会污染训练代码

## 结论

🟡 **脚本基本正确，但实验封装不完整**

`exp090` 的核心脚本思路是成立的，但还缺一套正式的实验记录与可追溯结果。
