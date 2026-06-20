# 通宵 pose+CLIP 深度融合涨点搜索 (2026-06-20 夜)

## 目标
找一个 pose+CLIP 融合机制, test.py >59.8 且清噪声(>~+0.5)。基线: exp341 CLIP=59.8, baseline=57.6, exp342外挂=60.0(+0.2 marginal)。

## 机制洞察(指导所有设计, 来自白天 A/B/C 全负)
exp341 +2.2 来自 **global 对齐纯 ID 文本原型**(把 global 塑成纯 ID-判别)。三个坑:
- A(pose-bias 池化, learnable query/k_proj)→ 通路吸收 i2t/t2i 梯度 → 57.6
- B(pose 调制 prompt)→ 原型 pose-aware → global 编码姿态非纯ID → 57.6
- C(K 部位对齐 ID 原型, learnable query)→ 部分吸收 → 58.0
**绕坑原则**: 姿态只帮"对齐信号干净", 不碰对齐目标(纯ID), 用无参数池化(不吸收)。

## 通宵 bet 矩阵(全双审查通过)
| | 机制 | 假设 | 状态 |
|---|---|---|---|
| b exp342b | un-detach LGPA(姿态塑造backbone,部位独立ID监督) | 塑造≠竞争对齐 → 破冗余 | 4090跑 |
| c exp342c | clean global 1.0x(GLOBAL_LOSS_SCALE 2.0 修M1) | 干净global → external从+0.2到+0.4 | 3090跑 |
| exp347 | param-free de-occluded 对齐(无参数池化) | 梯度直进backbone(修A吸收)+对齐去遮挡global到纯ID | 排4090 |
| exp348 | exp347 + occluder repulsion | 显式分离 可见=ID/遮挡=非ID → GAP更干净 | 排3090 |
| exp349 | Swin-Small全pose(73.2)+CLIP | 容量大+互补 → 组合>73.2 | 排4090 |

## 备用角度(若上述全 marginal)
- pose-reliability-weighted 对比: 高可见样本在 i2t/t2i 权重大 → 原型更干净(不碰feature,不编码姿态)
- CLIP-supervised pose parts 当独立描述子(per-part 学习原型, 不进 global 对齐 → 不吸收)
- pose-guided 遮挡增强 + CLIP 一致性(已部分在 exp349 的 PLBOA+CLIP)

## 结果(待填)
