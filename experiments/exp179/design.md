# 实验 exp179: SupCon T=0.05 on base architecture (PSG@Stage3 only, no PAPE)

## 动机
- SupCon 在 triple injection (exp176) 和 PAPE-only (exp177) 上都有正效果
- 最终消融：SupCon 在最基础的架构上（PSG@Stage3 only, no PAPE, no multi-stage）是否有效？
- 如果有效 → SupCon 是独立于架构的通用改善
- 如果无效 → SupCon 依赖 PAPE/multi-stage PSG

## 技术方案
- 基于 exp166 配置（PSG@Stage3, per-token, PLBOA）
- 增加 POSE_STR_SUPCON: True, POSE_STR_SUPCON_TEMP: 0.05
- 不使用 PAPE 或 multi-stage PSG

## 对照组
- exp166 (CE): 63.1/73.9
- exp177 (SupCon+PAPE): 63.5/75.3
