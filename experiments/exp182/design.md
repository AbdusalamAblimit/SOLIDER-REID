# 实验 exp182: SupCon + CE Joint Training (不替代，而是叠加)

## 动机
- exp174-179 证明 SupCon 替代 per-token CE 是核心突破
- 但 SupCon 和 CE 优化不同目标：SupCon 优化 metric space，CE 优化 classification boundary
- 如果两者联合使用（不替代），可能兼得两方面的优势
- 当前：global CE + per-token SupCon + per-token triplet
- 提议：global CE + per-token CE + per-token SupCon + per-token triplet

## 技术方案
- 修改 make_loss.py: SupCon 不替代 CE，而是作为额外 loss 叠加
- part_loss = CE + lambda * SupCon
- lambda = 0.5 (SupCon 权重)
- 基于 exp176 配置

## 对照组
- exp176 (SupCon only, T=0.05): 64.1/75.5
