# exp075 Multi-Seed 验证审查报告

## 审查范围
- `experiments/exp075/design.md`
- `experiments/exp075/monitor.md`
- `train.py`
- `configs/occluded_duke/pose_psg_gcn_paa.yml`
- `configs/occluded_duke/pose_psg_gcn_paa_roa.yml`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | MEDIUM | exp075 目录 | `exp075` 没有独立的本地 config 快照，更多是一次多 seed 运行记录。复现依赖“沿用 exp066/exp067 config 并手动改 seed”这一过程，而不是 exp075 目录内的自包含文件 | 未修复 |

## 审查通过项

- `train.py` 确实会把 `cfg.SOLVER.SEED` 同时作用到 `torch / cuda / numpy / random`
- exp075 没有新增模型代码或 loss 代码，因此不存在新的实现风险
- 监控记录清楚表明它是在复用 exp066 / exp067 的实现做 seed 验证

## 结论

🟡 **过程正确，但实验封装不完整**

`exp075` 不是一个新的代码实验，而是一个多 seed 复现实验。seed 机制本身没有问题，但文档封装不够自包含，后续若要让别人重跑，最好补一份明确的 seed-config 快照。
