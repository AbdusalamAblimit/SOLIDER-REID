# 实验 exp184: SupCon on GCN branch (non-STD-PR)

## 动机
- SupCon 在 STD-PR per-token 上是突破性发现 (+3.9/+2.1 on base)
- 问题：SupCon 是 specific to STD-PR per-token 还是通用的？
- 本实验测试 SupCon 在 GCN branch（pooled part feature）上的效果
- 如果有效 → SupCon 是通用改进，不依赖 per-token 结构
- 如果无效 → SupCon 依赖 per-token 的 diversity 结构

## 技术方案
- 基于 GCN + PLBOA config (pose_psg_gcn_plboa.yml)
- 增加 POSE_STR_SUPCON: True, POSE_STR_SUPCON_TEMP: 0.05
- GCN 返回 [global_feat, gcn_pooled] (2 elements)
- 代码修改: make_loss.py 条件从 len>3 放宽到 len>1，让 GCN 也能走 SupCon 路径

## 对照组
- exp030a-eq (GCN + PLBOA + CE, 3-seed): 60.73/72.57 (equal_concat)
- exp179 (STD-PR + SupCon + PLBOA): 64.2/74.9
