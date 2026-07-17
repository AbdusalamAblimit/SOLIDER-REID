# 实验 exp219: PACI WITHOUT OA-SD — Tiny

## 动机
- exp218 (PACI + OA-SD): `61.9/74.2` — 低于 OA-SD-only `63.2/75.4`
- 所有 "extra loss on top of OA-SD" 实验都表现不佳
- 假设: PACI 的 consistency loss 与 OA-SD 的 distillation loss 在 GCN 上梯度竞争
- **如果 PACI-only (无 OA-SD) 能超过 baseline (60.7)，PACI 仍有论文价值**

## 核心假设
PACI 的 per-ID per-part prototype consistency loss 作为 GCN 唯一额外训练信号
（无 OA-SD 竞争），可能超过 plain baseline (60.7)。

## 技术方案
- 与 exp218 相同但: POSE_OA_SD=False
- PACI weight=0.5, warmup=5

## 对照组
- exp030a baseline (no OA-SD, no PACI): 60.7
- exp191 OA-SD only: 63.2/75.4
- exp218 PACI + OA-SD: ~62 (预计)
