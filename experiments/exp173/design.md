# 实验 exp173: Triple Pose Injection

## 动机
- PAPE 1×1 (exp171) 是唯一正向的方向 (+0.1/+0.4)
- PAPE 3×3 (exp172) 不优于 1×1 — 更大 kernel 不帮忙
- 下一个维度：更多注入点。当前 PSG 只在 Stage 3，代码已支持多 Stage

## 核心假设
三层 pose injection 让整个 backbone 全程 pose-aware：
- PAPE: 输入层 (Conv2d 1×1, 1.7K params)
- PSG@Stage2: 中间层 (6 blocks × ~26K params = ~156K)  [Stage 2 = depths[2] = 6]
- PSG@Stage3: 高层 (2 blocks × ~51K params = ~102K)
- 总新增 ~156K params (Stage 2 PSG)

## 技术方案
- `POSE_PSG_STAGES: [2, 3]` (vs exp171 的 [-1] = Stage 3 only)
- `POSE_PATCH_EMBED: True` (保留 PAPE)
- 其余设置与 exp171 完全相同

## 对照组
- exp171 (PAPE + PSG@Stage3): 63.2/74.3
- 消融变量：增加 PSG@Stage2
