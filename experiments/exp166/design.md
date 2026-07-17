# 实验 exp166: STD-PR + Per-Token Classification

## 动机
STD-PR V1 的 6 tokens mean-pool 后 diversity 不够（所有 token 可能学到相似的东西）。
Per-token classification 强制每个 token 独立有身份判别力。

## 技术方案
- 每个 structural token 独立过一个 shared classifier
- 6 个 per-token CE loss + 1 个 pooled CE loss
- per-token CE weight = 0.5（不要太大，避免碎片化梯度）
- test-time: 6 tokens L2-normalize 后 concatenate（3×768=4608-d）
  注：实际可能需要降维到 128-d per token，否则 4608-d 太长

## 与现有方案的关系
- V1 mean-pool: 6 tokens → mean → 768-d → 1 CE (当前最佳)
- **V1 per-token**: 6 tokens → 各自 CE + concat test (本实验)

## 预期
- 每个 token 更有判别力 → 可能改善 R1（concatenation 保留更多信息）
- 但 4608-d 的 test feature 可能太稀疏 → 需要降维
