# 实验 exp175: PSG at ALL Stages [0,1,2,3] (无 PAPE)

## 动机
- exp173 triple injection (PAPE + PSG@[2,3]) = R1 74.7% (新高)
- 问题：PAPE 和 PSG@Stage2 哪个贡献更大？还是四重 PSG 更好？
- 本实验测试 PSG at ALL 4 stages，不用 PAPE，isolate PSG 多 stage 效果

## 技术方案
- `POSE_PSG_STAGES: [0, 1, 2, 3]` — 所有 Swin 阶段都注入 PSG
- `POSE_PATCH_EMBED: False` — 不使用 PAPE
- Stage 0: 96-d, 2 blocks → ~7K × 2 = 14K params
- Stage 1: 192-d, 2 blocks → ~14K × 2 = 28K params
- Stage 2: 384-d, 6 blocks → ~26K × 6 = 156K params
- Stage 3: 768-d, 2 blocks → ~51K × 2 = 102K params
- 总 PSG 参数: ~300K（vs exp166 只有 Stage 3 的 ~102K）

## 对照组
- exp173 (PAPE + PSG@[2,3]): 63.0/74.7
- exp166 (PSG@[3] only): 63.1/73.9
