# 论文素材管理

## 目录结构

```
experiments/paper_materials/
├── figures/
│   ├── method_overview/      # 方法总览图素材
│   ├── qualitative/          # 检索结果、attention 可视化
│   ├── tsne/                 # t-SNE 特征分布
│   └── ablation_charts/      # 消融实验图表
├── tables/
│   ├── main_results.md       # 与 SOTA 对比
│   ├── ablation.md           # 消融实验
│   └── efficiency.md         # 计算效率对比
└── story.md                  # 论文故事线（持续更新）
```

## story.md 模板

从第一个实验开始维护：

```markdown
# 论文故事线

## 暂定标题
## Motivation（为什么做这个）
- 现有问题 / 现有方法不足 / 我们的洞察

## 核心贡献（预计 3 点）
## 方法概述
## 实验证据链
- 实验 A 证明了 ... / 消融 B 证明了 ... / 可视化 C 展示了 ...

## 与 SOTA 对比 narrative
## 待补充的实验
```

## 迭代策略

- Round 1: 探索性实验 — 快速试错，20-30 epoch 看趋势
- Round 2: 创新点验证 — 消融实验、对比实验，证明每个组件必要
- Round 3: 完善补充 — 超参敏感性、可视化、效率分析

## Phase 1 论文学习（已完成）

12/12 论文已学习完毕，笔记在 `experiments/paper_notes/`。
创新分析在 `experiments/innovation_brainstorm.md` 和 `experiments/module_candidates.md`。
如需学习新论文，参考已有笔记格式。
