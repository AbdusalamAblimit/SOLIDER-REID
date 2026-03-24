# 实验 exp172: PAPE 3×3 (更大感受野)

## 动机
exp171 PAPE 1×1 是唯一产生正增益的方向 (+0.1/+0.4)。3×3 conv 可以捕捉局部 pose spatial pattern（相邻关键点的关系），可能进一步改善。

## 技术方案
- Conv2d(17, 96, 3×3, padding=1) 替代 Conv2d(17, 96, 1×1)
- 14,784 params (vs 1×1 的 1,728)
- 零初始化，其余设置与 exp171 完全相同

## 对照组
- exp171 (PAPE 1×1): 63.2/74.3
- 消融变量：仅改 POSE_PATCH_EMBED_KS: 1 → 3
