# 实验 exp203: Swin-Small + GCN+PAA+ROA + SupCon + PLBOA + 3-view

## 动机
- 4090 PAA log: Swin-Small + GCN+PAA+ROA (旧方法) = **70.8/81.7**
- exp202b: Swin-Small + STD-PR+SupCon+3-view (新方法) = **69.3/80.2**
- 旧方法在 Small 上反而高 1.5%！说明 GCN+PAA 架构在 Small 上更强
- **本实验**: 在 GCN+PAA 基础上加 SupCon + PLBOA + 3-view，试图超过 70.8

## 核心假设
GCN+PAA+ROA 架构 (70.8) + SupCon+PLBOA+3-view 的训练改进 → 可能达到 72-73%

## 技术方案
- 基于 `pose_psg_gcn_paa_roa.yml` 配置
- 改 backbone 为 Swin-Small
- 加 SupCon, PLBOA, 3-view parallel aug
- WITH_CP=True (3-view Small 需要)
- LR=0.0004

## 对照组
- 4090 PAA (GCN+PAA+ROA, CE): 70.8/81.7
- exp202b (STD-PR+SupCon+3-view): 69.3/80.2
