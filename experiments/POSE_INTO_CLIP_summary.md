# 姿态融进 CLIP-ID-prompt 机制 — 三方系统验证 (2026-06-20)

## 背景
exp341: CLIP-ReID 可学习 ID prompt 对齐 raw global → **+2.2** (59.8 vs 57.6 baseline)。
用户路线: 把姿态融进这个能涨的 CLIP 机制, 再涨。

## 三种融法 (全 e120, test.py global, 单变量 vs exp341, 双审查通过)
| | 机制 | global | vs exp341 | 结论 |
|---|---|---|---|---|
| A exp343 | pose-bias 池化特征替代 global 做对齐 | 57.6 | -2.2 | 吸收陷阱(全) |
| B exp344 | pose 调制 prompt context(zero-init) | 57.6 | -2.2 | global 编码姿态稀释纯ID |
| C exp345 | K=3 pose-localized 部位各对齐 ID 原型 | 58.0 | -1.8 | 吸收陷阱(部分) |

## 核心洞察
exp341 涨点来自**纯 ID 对齐**: raw global ↔ 纯 ID 文本原型, 把 global 塑造成纯 ID-判别。
**姿态以任何"融进对齐"的方式进来, 都在与这个纯 ID 对齐竞争或稀释它** → 掉回 baseline。
- A/C: 姿态建新通路(有自己参数), 吸收 i2t/t2i 梯度, backbone/global 拿不到直接对齐。
- B: 姿态进 prompt → 原型 pose-aware → global 被拉去编码姿态而非纯 ID。

## 剩余未测: 姿态当独立描述子 (exp342, 不融进对齐, 拼接)

## 全口径最终结果 (e120, test.py, mAP / Rank-1)
| | pose 怎么加 | 描述子 | mAP | R1 |
|---|---|---|---|---|
| exp341 (Step1 纯 CLIP) | — | global | 59.8 | 68.4 |
| A exp343 | 换对齐的特征 | global | 57.6 | 65.8 |
| B exp344 | 调制 prompt | global | 57.6 | 66.4 |
| C exp345 | 部位对齐 | global | 58.0 | 67.9 |
| **exp342 (外挂/分离)** | LGPA 部位另算拼接 | equal_concat | **60.0** | **68.9** |

exp342 global=59.6 → equal_concat 60.0 = pose 净 +0.4 mAP (同尺度)。

## 终结论
- 整合式 (姿态进对齐): 3/3 全负 (-1.8~-2.2 mAP), 姿态与纯 ID 对齐竞争。
- 外挂式 (姿态当独立描述子, 不碰对齐): +0.2 mAP / +0.5 R1, **seed 噪声内, marginal**。
- **Step1 CLIP 机制 (+2.2) 是真贡献; 姿态在其上最多 marginal, 且必须"外挂不碰对齐"**。用户路线: Step1 成, Step2 系统证明姿态加不动 (整合害, 外挂微涨)。
