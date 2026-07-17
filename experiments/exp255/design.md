# exp255: Small LGPA-D + GCN hidden 512 + 2-stage PSG + OA-SD

## 动机
目标: 推 Small mAP 向 75%。当前最佳 exp249 MaxSim = 73.3。
GCN hidden 256 → 512: 增加结构分支的表征容量。
同时用 2-stage PSG (exp254 在 Tiny 上 R1 最优配置)。

## 核心假设
更大的 GCN 隐藏层能更好地建模 skeleton 结构关系，提供更丰富的结构特征。

## 技术方案
- POSE_GCN_HIDDEN: 256 → 512 (单变量 vs exp254b)
- POSE_PSG_STAGES=[-2,-1] (2-stage PSG)
- 其余与 exp249 相同: Small + LGPA-D + GCN + OA-SD + PLBOA

## 参数增量
- GCN Layer 1: Linear(768→512) + Linear(512→512) → +512*768 + 512*512 = 655K params
- GCN Layer 2: Linear(512→512) + Linear(512→512) → +524K params  
- Total: ~1.2M (vs 256 hidden: ~0.6M)
- 可忽略的额外内存

## 对照组
- exp254b (Small 2-stage PSG, GCN 256): 进行中
- exp249 (Small 1-stage PSG, GCN 256): 71.9/81.8, MaxSim 73.3/83.2

## 预期结果
- 成功: 72.5+ mAP, MaxSim 74+ (向 75% 迈进)
- 中性: ≈ exp249
- 失败: < exp249 (过拟合)
