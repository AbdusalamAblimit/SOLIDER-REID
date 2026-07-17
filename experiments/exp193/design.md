# 实验 exp193: OA-SD + 3-view Parallel Aug + CE

## 动机
- exp191 (OA-SD + CE) = 63.2/75.4 — OA-SD 在 CE 下独立有效 (+2.9/+2.6 vs CE base)
- exp190 (3-view + CE) — 正在跑，ep40=60.3/73.2（趋势良好）
- 两个方向都在 CE 下有效，组合是否 additive/superlinear？
- 如果组合后超过 exp187 (64.9/76.6)，说明 CE+OA-SD+3-view 可能是比 SupCon 更好的训练范式

## 核心假设
OA-SD (occlusion invariance via distillation) 和 3-view parallel augmentation (更多数据多样性) 是正交的改进方向，组合应产生 additive 增益。

## 技术方案
- 配置 = exp190 + POSE_OA_SD=True
- 即: 3-view parallel aug + CE + PLBOA + OA-SD (EMA teacher)
- 代码上无需修改：OA-SD 已支持 parallel_aug path（2-view list 在 parallel_aug 的每个 view 中）
- **注意**: 需验证 OA-SD 和 parallel_aug 的交互——OA-SD 给每个 view 做 2-view (student+teacher)，而 parallel_aug 有 3 个 view。总共 3×2=6 个 forward？
- **如果显存不够**: OA-SD 只对主 view (occluded view) 做 distillation，不对 ROA/heavy 做

## 预期结果
- 假设成立: mAP 65-66%, R1 77-78% (超过 exp187)
- 如果失败: 可能 OA-SD 在 3-view 下冗余（3-view 本身提供了多样性）

## 对照组
- exp190 (3-view + CE, 无 OA-SD): 预计 ~63-64% (跑完中)
- exp191 (OA-SD + CE, 1-view): 63.2/75.4
- exp187 (3-view + SupCon): 64.9/76.6

## 关键问题
1. 显存: 3-view × 2-view (OA-SD) = 6 forward passes。可能 OOM on 3090 (24GB)
2. 如果 OOM: 只对 PLBOA view 做 OA-SD distillation (其他 view 不做 teacher forward)
