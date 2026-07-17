# 实验 exp206: Swin-Small + GCN+PAA + CE + OA-SD (远程 1-view)

## 动机
- 4090 PAA (GCN+PAA+CE): 70.8/81.7 — GCN 架构在 Small 上最强
- OA-SD 在 Tiny CE 路线: +2.9/+2.6 (exp191 vs exp166r)
- **假设**: OA-SD 在 Small GCN+PAA 上也能提供 +2-3% → 72-73%+

## 核心假设
OA-SD (EMA teacher distillation) 在 Swin-Small GCN+PAA CE 路线上也有效。

## 技术方案
- 基于 pose_psg_gcn_paa_roa.yml + Small backbone + OA-SD + PLBOA
- 无 SupCon（CE 路线，OA-SD 与 CE 兼容）
- 远程 16GB 1-view

## 对照组
- 4090 PAA (GCN+PAA, CE, no OA-SD): 70.8/81.7
- exp191 (Tiny, OA-SD+CE): 63.2/75.4
