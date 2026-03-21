# 实验 exp142: SKC（Support-Conditioned Keypoint Completion）

## 2026-03-21 19:02 启动前记录

- 状态：仅完成设计，未改代码，未启动训练
- 当前定位：本地主线从 `LPCS` 小变体切换到 feature-space support completion 大改动
- 原因：
  1. `exp109` 已说明真正 headroom 在 `support incomplete`
  2. `exp119-140` 的 pair correction 系列已经提供了足够多“机制有用但突破有限”的证据
  3. 用户明确要求不要继续围绕一个小点耗时间
- 当前判断：
  1. `exp141` 虽已完成二次审查，但仍属于 `LPCS` 家族增量线，暂不启动
  2. `exp142` 将作为本地下一条真正不同的创新点
  3. 下一步先改代码，再做全面 Claude 审查

## 预设监控清单

后续真正启动训练后，每次检查除了常规 `loss / mAP / R1`，还必须补以下行为日志：

- `skc_lmr`
- `skc_spr`
- `skc_arr`
- `skc_gm`
- `skc_gs`
- `skc_dn`
- `skc_pc`
- `skc_pcnt`
- `skc_cl`
- `skc_pre`
- `skc_post`

如果启动后这些日志缺失，则本次 run 视为不可解释 run，需要优先补日志再继续。
