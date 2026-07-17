# 实验 exp121: SCRD Freeze-30

## 动机

- `exp120` 正在验证：support-complete teacher 是否能把 `exp109` 的 headroom 接到 `exp119` 的 relational distillation 上
- 但 `SCKD` 系列早就暴露过一个问题：**online teacher 会持续变硬**
- 因此如果 `exp120` 成立，下一层最关键的问题不是扫 loss 权重，而是：
  **support-complete relational teacher 是否也需要稳定化**

远程并行最合适的单变量就是：
- 保持 `exp120` 完全不变
- 只把 `POSE_CSRD_ST_UPDATE_STOP_EPOCH` 设为 `30`

## 核心假设

1. 如果 `exp120` 的主要风险也来自 online teacher non-stationarity，那么在 support-complete bank 已初步成熟后冻结更新，可能会比持续在线更新更稳
2. `30` 比 `20` 更合理，因为：
   - `CSRD` 本身在 `epoch > 20` 才激活
   - 让 teacher 多积累约 10 个 epoch，再冻结，更像“成熟后固定”
3. 若 freeze30 优于 exp120，说明后续主方法应把“support-complete”与“stable teacher”一起写

## 技术方案

- 基础配置：`exp120`
- 唯一改动：
  - `POSE_CSRD_ST_UPDATE_STOP_EPOCH: 30`
- 其它全部保持不变：
  - `POSE_CSRD_SUPPORT_TEACHER=True`
  - `POSE_CSRD_ST_UPDATE_THR=0.7`
  - `POSE_CSRD_WEIGHT=0.5`
  - `POSE_CSRD_TAU=0.10`

## 对照组

- 本地主实验：`exp120 SCRD`（online support-complete teacher）
- 远程对照：`exp121 SCRD freeze30`

## 预期结果

- 若 freeze30 更好：
  - 说明 support-complete teacher 也存在 non-stationary / hardening 问题
  - 下一步应继续沿“stable support-complete relational teacher”写主方法
- 若 freeze30 与 exp120 等价或更差：
  - 说明当前优先级仍是 teacher completeness，而不是 stability

## 风险与失败解释

1. 远程 5060 Ti 较慢，观测会滞后
2. 若 `exp120` 自身后续不成立，则 `exp121` 的意义也会下降
3. 冻结过早可能让 bank 还没积累够 support，导致 teacher 反而变弱
