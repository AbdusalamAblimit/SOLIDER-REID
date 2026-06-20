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
