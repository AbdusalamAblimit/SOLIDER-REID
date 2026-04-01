# 实验 exp209: Small + STD-PR + CE + OA-SD (远程 1-view)

## 动机
- OA-SD 在 GCN+CE 路线 = +2.9/+2.6 (Tiny), +0 (Small vs 4090 PAA)
- OA-SD 在 STD-PR+SupCon 路线 = 负向 (互斥)
- **但 OA-SD + STD-PR + CE 从未测试过！**
- 如果 OA-SD 在 STD-PR+CE 上也有效: STD-PR+CE base ~65.8 + OA-SD +2-3 = **68-69%**
- 这与 GCN+PAA+OA-SD (70.5) 比较，确认哪个架构+OA-SD 更好

## 核心假设
OA-SD 在 CE 路线下与架构无关——在 STD-PR 上也应该有效。

## 技术方案
- 配置: pose_psg_stdpr_pertoken_plboa_pape_ms_supcon_small.yml
- 关闭 SupCon: POSE_STR_SUPCON False
- 启用 OA-SD: POSE_OA_SD True
- 1-view (远程 16GB)

## 对照组
- exp202 (STD-PR+SupCon 1v Small): 67.9/79.5
- exp206 (GCN+PAA+OA-SD 1v Small): 70.5/82.3
