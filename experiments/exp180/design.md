# 实验 exp180: SupCon T=0.05 + PLBOA gradient mode

## 动机
- SupCon T=0.05 是最佳 trade-off (64.1/75.5)
- PLBOA 当前用 'lower' mode（只遮挡 hip 以下区域）
- 'gradient' mode 用 bottom-heavy 概率分布，从底部到顶部递减
- gradient mode 提供更多样的遮挡高度 → 可能与 SupCon 的 hard negative mining 更好协同

## 技术方案
- 基于 exp176 配置 (best: triple + SupCon T=0.05)
- 仅改 POSE_LOWER_BODY_OCC_MODE: 'lower' → 'gradient'

## 对照组
- exp176 (SupCon T=0.05 + lower mode): 64.1/75.5
