# exp251: Multi-Stage PSG + PAA on Tiny LGPA-D+GCN

## 动机
PSG 目前只注入 Stage 3 (+1.7% mAP)。将 pose 注入扩展到 Stage 2+3 让 backbone 更早获得 pose 信息。
PAA (Pose Additive Adapter) 提供加法 pose 信号，与 PSG 乘法门控互补。
exp073 测试过 multi-stage PAA (-0.5 mAP)，但 multi-stage PSG 从未测试过。

## 核心假设
Stage 2 的中层特征也能受益于 pose 调制；PSG+PAA 双模式互补。

## 技术方案
- 仅 config 变更：POSE_PSG_STAGES=[-2,-1], POSE_ADDITIVE_ADAPTER=True
- 其余与 exp246b 相同：Tiny LGPA-D+GCN+OA-SD+PLBOA

## 代码修改
无代码修改，仅 config 参数。

## 对照组
- exp246b (Tiny LGPA-D+GCN, Stage3-only PSG): 65.5/77.2, MaxSim 66.3/77.7

## 预期结果
- 成功: +0.5% mAP (multi-stage PSG 有效)
- 中性: ±0.2% (Stage 2 信息冗余)
- 失败: -1% (早期 pose 注入干扰)
