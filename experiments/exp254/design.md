# exp254: 2-Stage PSG (Stage2+3, 无 PAA) on Tiny LGPA-D+GCN

## 动机
补全 PSG 消融表。已有: 1-stage (exp246b), 2-stage+PAA (exp251), 3-stage (exp253)。
缺少: 2-stage 无 PAA。分离 multi-stage PSG vs PAA 的贡献。

## 核心假设
2-stage PSG (无 PAA) 与 exp253 (3-stage) 和 exp246b (1-stage) 效果相当。

## 技术方案
- POSE_PSG_STAGES=[-2,-1], POSE_ADDITIVE_ADAPTER=False
- 其余与 exp246b 相同

## 对照组
- exp246b (1-stage): 65.5/77.2, MaxSim 66.3/77.7
- exp251 (2-stage+PAA): 65.2/76.2, MaxSim 65.9/76.8
- exp253 (3-stage): 65.1/76.2, MaxSim 66.0/77.2
