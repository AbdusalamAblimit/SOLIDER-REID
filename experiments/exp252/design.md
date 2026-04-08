# exp252: Multi-Stage PSG + PAA on Small LGPA-D+GCN

## 动机
与 exp251 相同假设，在 Small backbone 上验证。
Small 有 18 个 Stage 3 blocks + 2 个 Stage 2 blocks。

## 核心假设
Multi-stage PSG + PAA 在更大 backbone 上同样有效或更有效。

## 技术方案
- 仅 config 变更：POSE_PSG_STAGES=[-2,-1], POSE_ADDITIVE_ADAPTER=True
- 其余与 exp249 相同：Small LGPA-D+GCN+OA-SD+PLBOA+WITH_CP

## 代码修改
无代码修改，仅 config 参数。

## 对照组
- exp249 (Small LGPA-D+GCN, Stage3-only PSG): 71.9/81.8, MaxSim 73.3/83.2

## 预期结果
- 成功: 72+ mAP, MaxSim 74+
- 中性: ≈ exp249
- 失败: < exp249 (multi-stage 在 Small 上有冲突)
