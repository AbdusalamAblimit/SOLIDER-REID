# Paper 20: PADE — Parallel Augmentation and Dual Enhancement
**来源**: ICASSP 2024
**仓库**: https://github.com/ziwang1121/PADE
**核心思想**: 三路并行增强训练（原图 + crop遮挡 + 强制erasing）

## 核心创新
1. **Parallel Augmentation**: 同一图片生成3个增强版本同时训练
   - img1: 标准resize
   - img2: RandomResizedCrop(pad30 + random crop) — 模拟遮挡
   - img3: 100% Random Erasing — 强制部分可见
2. **Dual Enhancement**: 50-50 加权 local(JPM) + occlusion(三路) branches

## 可移植到我们框架的关键机制
1. **RandomResizedCrop 遮挡模拟** — 比 ROA 更简单，不需要外部数据
2. **三路训练范式** — 但训练时间 3x，需要评估性价比

## 对我们框架的启发
- PADE 证明"多样化遮挡增强"是有效的方向
- 我们的 ROA 已经在这个方向上，但只有一种增强
- 可能的组合: ROA + RandomResizedCrop = 更多样化的遮挡训练

## 局限性
- 三路训练 3x 时间开销
- 没有使用 pose 信息
- JPM 的 local parts 是随机切分，不是 pose-guided
